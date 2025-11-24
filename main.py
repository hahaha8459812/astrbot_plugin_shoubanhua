import asyncio
import base64
import functools
import io
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import aiohttp
from PIL import Image as PILImage

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import At, Image, Reply, Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent


PRESET_MODELS = [
    "nano-banana",
    "nano-banana-2-4k",
    "nano-banana-2-2k",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
    "nano-banana-hd",
    "gemini-2.5-flash-image-preview"
]


@register(
    "astrbot_plugin_shoubanhua",
    "shskjw",
    "Google Gemini 手办化/图生图插件",
    "1.5.0",
    "https://github.com/shkjw/astrbot_plugin_shoubanhua",
)
class FigurineProPlugin(Star):
    
    class ImageWorkflow:
        def __init__(self, proxy_url: str | None = None, max_retries: int = 3):
            if proxy_url:
                logger.info(f"ImageWorkflow 使用代理: {proxy_url}")
            self.proxy = proxy_url
            self.max_retries = max_retries

        async def _download_image(self, url: str) -> bytes | None:
            logger.info(f"正在下载图片: {url} (重试上限: {self.max_retries})")
            
            for i in range(self.max_retries + 1):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, proxy=self.proxy, timeout=60) as resp:
                            resp.raise_for_status()
                            return await resp.read()
                except Exception as e:
                    if i < self.max_retries:
                        logger.warning(f"图片下载失败 (第{i+1}次): {e}, 1秒后重试...")
                        await asyncio.sleep(1)
                    else:
                        logger.error(f"图片下载最终失败: {url}, 错误: {e}")
                        return None
            return None

        async def _get_avatar(self, user_id: str) -> bytes | None:
            if not user_id.isdigit():
                return None
            
            avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
            return await self._download_image(avatar_url)

        def _extract_first_frame_sync(self, raw: bytes) -> bytes:
            img_io = io.BytesIO(raw)
            try:
                with PILImage.open(img_io) as img:
                    if getattr(img, "is_animated", False):
                        img.seek(0)
                    
                    first_frame = img.convert("RGBA")
                    out_io = io.BytesIO()
                    first_frame.save(out_io, format="PNG")
                    return out_io.getvalue()
            except Exception:
                pass
            return raw

        async def _load_bytes(self, src: str) -> bytes | None:
            raw: bytes | None = None
            loop = asyncio.get_running_loop()
            
            if Path(src).is_file():
                raw = await loop.run_in_executor(None, Path(src).read_bytes)
            elif src.startswith("http"):
                raw = await self._download_image(src)
            elif src.startswith("base64://"):
                raw = await loop.run_in_executor(None, base64.b64decode, src[9:])
            
            if not raw:
                return None
                
            return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)

        async def get_images(self, event: AstrMessageEvent) -> List[bytes]:
            img_bytes_list: List[bytes] = []
            at_user_ids: List[str] = []

            for seg in event.message_obj.message:
                if isinstance(seg, Reply) and seg.chain:
                    for s_chain in seg.chain:
                        if isinstance(s_chain, Image):
                            if s_chain.url:
                                img = await self._load_bytes(s_chain.url)
                                if img:
                                    img_bytes_list.append(img)
                            elif s_chain.file:
                                img = await self._load_bytes(s_chain.file)
                                if img:
                                    img_bytes_list.append(img)

            for seg in event.message_obj.message:
                if isinstance(seg, Image):
                    if seg.url:
                        img = await self._load_bytes(seg.url)
                        if img:
                            img_bytes_list.append(img)
                    elif seg.file:
                        img = await self._load_bytes(seg.file)
                        if img:
                            img_bytes_list.append(img)
                elif isinstance(seg, At):
                    at_user_ids.append(str(seg.qq))

            if img_bytes_list:
                return img_bytes_list

            if at_user_ids:
                for user_id in at_user_ids:
                    avatar = await self._get_avatar(user_id)
                    if avatar:
                        img_bytes_list.append(avatar)
                return img_bytes_list

            avatar = await self._get_avatar(event.get_sender_id())
            if avatar:
                img_bytes_list.append(avatar)

            return img_bytes_list

        async def terminate(self):
            pass

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.plugin_data_dir = StarTools.get_data_dir()
        
        self.user_counts_file = self.plugin_data_dir / "user_counts.json"
        self.user_counts: Dict[str, int] = {}
        
        self.group_counts_file = self.plugin_data_dir / "group_counts.json"
        self.group_counts: Dict[str, int] = {}
        
        self.user_checkin_file = self.plugin_data_dir / "user_checkin.json"
        self.user_checkin_data: Dict[str, str] = {}
        
        self.prompt_map: Dict[str, str] = {}
        
        self.key_index = 0
        self.key_lock = asyncio.Lock()
        self.iwf: Optional[FigurineProPlugin.ImageWorkflow] = None

    async def initialize(self):
        use_proxy = self.conf.get("use_proxy", False)
        proxy_url = self.conf.get("proxy_url") if use_proxy else None
        retries = self.conf.get("download_retries", 3)
        
        self.iwf = self.ImageWorkflow(proxy_url, max_retries=retries)
        
        await self._load_prompt_map()
        await self._load_user_counts()
        await self._load_group_counts()
        await self._load_user_checkin_data()
        
        logger.info("FigurinePro 插件已加载")
        
        if not self.conf.get("api_keys") and not self.conf.get("custom_model_1_key") and not self.conf.get("custom_model_2_key"):
            logger.warning("FigurinePro: 未配置任何 API 密钥")

    async def _load_prompt_map(self):
        self.prompt_map.clear()
        prompt_list = self.conf.get("prompt_list", [])
        for item in prompt_list:
            try:
                if ":" in item:
                    key, value = item.split(":", 1)
                    self.prompt_map[key.strip()] = value.strip()
            except ValueError:
                pass

    def _get_all_models(self) -> List[str]:
        models = list(PRESET_MODELS)
        c1 = self.conf.get("custom_model_1", "").strip()
        c2 = self.conf.get("custom_model_2", "").strip()
        
        if c1:
            models.append(c1)
        if c2:
            models.append(c2)
            
        return models

    @filter.command("切换API模式", aliases={"SwitchApi"}, prefix_optional=True)
    async def on_switch_api_mode(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            yield event.plain_result("❌ 只有管理员可以执行此操作。")
            return
        
        current_mode = self.conf.get("api_mode", "generic")
        raw = event.message_str.strip()
        parts = raw.split()
        target_mode = parts[1].lower() if len(parts) > 1 else ""
        
        if not target_mode:
            msg = f"ℹ️ 当前 API 模式: **{current_mode}**\n"
            msg += "可选项:\n"
            msg += "1. `generic` (通用格式)\n"
            msg += "2. `gemini_official` (Gemini原生格式)\n"
            msg += "用法: `#切换API模式 <模式名>`"
            yield event.plain_result(msg)
            return
            
        if target_mode not in ["generic", "gemini_official"]:
            yield event.plain_result("❌ 模式无效。")
            return
            
        self.conf["api_mode"] = target_mode
        try: 
            if hasattr(self.conf, "save"):
                self.conf.save()
        except:
            pass
            
        yield event.plain_result(f"✅ API 模式已切换为: **{target_mode}**")

    @filter.command("切换模型", aliases={"SwitchModel", "模型列表"}, prefix_optional=True)
    async def on_switch_model(self, event: AstrMessageEvent):
        all_models = self._get_all_models()
        raw_msg = event.message_str.strip()
        parts = raw_msg.split()
        
        if len(parts) == 1:
            current_model = self.conf.get("model", "nano-banana")
            current_api_mode = self.conf.get("api_mode", "generic")
            
            msg = "📋 **可用模型列表**:\n"
            msg += "------------------\n"
            
            for idx, model_name in enumerate(all_models):
                seq_num = idx + 1
                status = "✅ (当前)" if model_name == current_model else ""
                is_custom = idx >= len(PRESET_MODELS)
                type_mark = " [自]" if is_custom else ""
                msg += f"{seq_num}. {model_name}{type_mark} {status}\n"
                
            msg += "------------------\n"
            msg += f"📡 **当前API模式**: {current_api_mode}\n"
            msg += "------------------\n"
            msg += "📝 **指令**:\n"
            msg += "1. `#切换模型 <序号>`\n"
            msg += "2. `#切换API模式 <模式名>`\n"
            msg += "3. `#手办化(序号) [图片]`"
            
            yield event.plain_result(msg)
            return

        arg = parts[1]
        if not self.is_global_admin(event):
            yield event.plain_result("❌ 只有管理员可以更改全局默认模型。")
            return

        if not arg.isdigit():
            yield event.plain_result("❌ 格式错误。")
            return
        
        target_idx = int(arg) - 1
        
        if 0 <= target_idx < len(all_models):
            new_model = all_models[target_idx]
            self.conf["model"] = new_model
            try:
                if hasattr(self.conf, "save"):
                    self.conf.save()
            except:
                pass
            yield event.plain_result(f"✅ 切换成功！\n当前默认模型: **{new_model}**")
        else:
            yield event.plain_result(f"❌ 序号无效。")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_figurine_request(self, event: AstrMessageEvent):
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command:
            return
            
        text = event.message_str.strip()
        if not text:
            return
        
        full_cmd_match = text.split()[0].strip()
        suffix_match = re.search(r"[\(（](\d+)[\)）]$", full_cmd_match)
        
        temp_model_idx = None
        cmd = full_cmd_match
        
        if suffix_match:
            temp_model_idx = int(suffix_match.group(1))
            cmd = full_cmd_match[:suffix_match.start()]
            
        bnn_command = self.conf.get("extra_prefix", "bnn")
        user_prompt = ""
        is_bnn = False
        
        if cmd == bnn_command:
            user_prompt = text.removeprefix(full_cmd_match).strip()
            is_bnn = True
            if not user_prompt:
                return
        elif cmd in self.prompt_map:
            user_prompt = self.prompt_map.get(cmd)
        else:
            return

        sender_id = event.get_sender_id()
        group_id = event.get_group_id()
        is_master = self.is_global_admin(event)

        if not is_master:
            if sender_id in self.conf.get("user_blacklist", []):
                return
            if group_id and group_id in self.conf.get("group_blacklist", []):
                return
            if self.conf.get("user_whitelist", []) and sender_id not in self.conf.get("user_whitelist", []):
                return
            if group_id and self.conf.get("group_whitelist", []) and group_id not in self.conf.get("group_whitelist", []):
                return
            
            user_count = self._get_user_count(sender_id)
            group_count = self._get_group_count(group_id) if group_id else 0
            user_limit_on = self.conf.get("enable_user_limit", True)
            group_limit_on = self.conf.get("enable_group_limit", False) and group_id
            
            has_group_count = not group_limit_on or group_count > 0
            has_user_count = not user_limit_on or user_count > 0
            
            if group_id:
                if not has_group_count and not has_user_count:
                    yield event.plain_result("❌ 次数已用尽。")
                    return
            elif not has_user_count:
                yield event.plain_result("❌ 次数已用尽。")
                return

        img_bytes_list = await self.iwf.get_images(event)
        if not img_bytes_list:
            if not is_bnn:
                yield event.plain_result("请发送或引用一张图片。")
                return
        
        images_to_process = []
        if is_bnn and len(img_bytes_list) > 5:
            images_to_process = img_bytes_list[:5]
        else:
            images_to_process = img_bytes_list
            
        if not is_bnn:
            images_to_process = [img_bytes_list[0]]
            
        display_cmd = user_prompt[:10] + '...' if is_bnn else cmd
        yield event.plain_result(f"🎨 收到请求，正在生成 [{display_cmd}]...")
        
        override_model_name = None
        all_models = self._get_all_models()
        
        if temp_model_idx is not None:
            if 1 <= temp_model_idx <= len(all_models):
                override_model_name = all_models[temp_model_idx - 1]
            else:
                yield event.plain_result(f"⚠️ 序号 {temp_model_idx} 无效。")
        
        if not is_master:
            if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0:
                await self._decrease_group_count(group_id)
            elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                await self._decrease_user_count(sender_id)

        start_time = datetime.now()
        res = await self._call_api(images_to_process, user_prompt, override_model=override_model_name)
        elapsed = (datetime.now() - start_time).total_seconds()

        if isinstance(res, bytes):
            caption_parts = [f"✅ 生成成功 ({elapsed:.2f}s)"]
            if is_master:
                caption_parts.append("剩余: ∞")
            yield event.chain_result([Image.fromBytes(res), Plain(" | ".join(caption_parts))])
        else:
            yield event.plain_result(f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {res}")
        
        event.stop_event()

    @filter.command("文生图", prefix_optional=True)
    async def on_text_to_image_request(self, event: AstrMessageEvent):
        raw_cmd = event.message_str.strip()
        prompt = raw_cmd
        override_model_name = None
        
        match = re.match(r"^[\(（](\d+)[\)）]\s*(.*)", prompt)
        if match:
            idx = int(match.group(1))
            prompt = match.group(2)
            all_models = self._get_all_models()
            if 1 <= idx <= len(all_models):
                override_model_name = all_models[idx-1]
            else:
                 yield event.plain_result(f"⚠️ 序号 {idx} 无效。")
                 return

        if not prompt:
            yield event.plain_result("请提供文生图描述。")
            return
        
        sender_id = event.get_sender_id()
        if not self.is_global_admin(event):
            if self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) <= 0:
                 yield event.plain_result("❌ 您的使用次数已用完。")
                 return

        info_str = f"🎨 收到文生图请求，正在生成 [{prompt[:10]}...]"
        if override_model_name:
            info_str += f" (模型: {override_model_name})"
        yield event.plain_result(info_str)

        if not self.is_global_admin(event):
            if self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                await self._decrease_user_count(sender_id)

        start_time = datetime.now()
        res = await self._call_api([], prompt, override_model=override_model_name)
        elapsed = (datetime.now() - start_time).total_seconds()

        if isinstance(res, bytes):
            yield event.chain_result([Image.fromBytes(res), Plain(f"✅ 生成成功 ({elapsed:.2f}s)")])
        else:
            yield event.plain_result(f"❌ 生成失败: {res}")
        
        event.stop_event()

    @filter.command("设置自定义key", aliases={"setk"}, prefix_optional=True)
    async def set_custom_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
            
        parts = event.message_str.strip().split()
        if len(parts) < 3:
            yield event.plain_result("格式错误。用法: #设置自定义key <1或2> <key>")
            return
        
        idx = parts[1]
        key_val = parts[2]
        
        if idx == "1":
            self.conf["custom_model_1_key"] = key_val
            msg = "✅ 自定义模型1 的 Key 已更新。"
        elif idx == "2":
            self.conf["custom_model_2_key"] = key_val
            msg = "✅ 自定义模型2 的 Key 已更新。"
        else:
            yield event.plain_result("❌ 仅支持设置 1 或 2。")
            return
        
        try:
            if hasattr(self.conf, "save"):
                self.conf.save()
        except:
            pass
            
        yield event.plain_result(msg)

    @filter.command("lm添加", aliases={"lma"}, prefix_optional=True)
    async def add_lm_prompt(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
            
        raw = event.message_str.strip()
        raw = re.sub(r'^[#\/]?(lm添加|lma)\s*', '', raw, flags=re.IGNORECASE).strip()
        
        if ":" not in raw:
            yield event.plain_result('格式错误, 示例: 触发词:提示词')
            return
            
        key, new_value = map(str.strip, raw.split(":", 1))
        prompt_list = self.conf.get("prompt_list", [])
        found = False
        
        for idx, item in enumerate(prompt_list):
            if item.strip().startswith(key + ":"):
                prompt_list[idx] = f"{key}:{new_value}"
                found = True
                break
        
        if not found:
            prompt_list.append(f"{key}:{new_value}")
            
        self.conf["prompt_list"] = prompt_list
        if hasattr(self.conf, "save"):
            self.conf.save()
            
        await self._load_prompt_map()
        yield event.plain_result(f"✅ 已保存预设:\n{key}:{new_value}")

    @filter.command("lm帮助", aliases={"lmh", "手办化帮助"}, prefix_optional=True)
    async def on_prompt_help(self, event: AstrMessageEvent):
        parts = event.message_str.strip().split()
        keyword = parts[1] if len(parts) > 1 else ""
        
        if not keyword:
            help_text = self.conf.get("help_text")
            if help_text:
                yield event.plain_result(help_text)
                return
            
            keys = sorted(list(self.prompt_map.keys()))
            yield event.plain_result(f"🎨 预设列表: {', '.join(keys) or '(无)'}")
            return
            
        prompt = self.prompt_map.get(keyword)
        if prompt:
            yield event.plain_result(f"📄 预设 [{keyword}] 内容:\n{prompt}")
        else:
            yield event.plain_result(f"❌ 未找到 [{keyword}]")

    def is_global_admin(self, event: AstrMessageEvent) -> bool:
        admin_ids = self.context.get_config().get("admins_id", [])
        return event.get_sender_id() in admin_ids

    async def _load_user_counts(self):
        if not self.user_counts_file.exists():
            self.user_counts = {}
            return
        try:
            content = await asyncio.to_thread(self.user_counts_file.read_text, "utf-8")
            self.user_counts = json.loads(content)
        except:
            self.user_counts = {}

    async def _save_user_counts(self):
        try:
            data = json.dumps(self.user_counts, indent=4)
            await asyncio.to_thread(self.user_counts_file.write_text, data, "utf-8")
        except:
            pass

    def _get_user_count(self, uid: str) -> int:
        return self.user_counts.get(str(uid), 0)

    async def _decrease_user_count(self, uid: str):
        count = self._get_user_count(uid)
        if count > 0:
            self.user_counts[str(uid)] = count - 1
            await self._save_user_counts()

    async def _load_group_counts(self):
        if not self.group_counts_file.exists():
            self.group_counts = {}
            return
        try:
            content = await asyncio.to_thread(self.group_counts_file.read_text, "utf-8")
            self.group_counts = json.loads(content)
        except:
            self.group_counts = {}

    async def _save_group_counts(self):
        try:
            data = json.dumps(self.group_counts, indent=4)
            await asyncio.to_thread(self.group_counts_file.write_text, data, "utf-8")
        except:
            pass

    def _get_group_count(self, group_id: str) -> int:
        return self.group_counts.get(str(group_id), 0)

    async def _decrease_group_count(self, group_id: str):
        count = self._get_group_count(group_id)
        if count > 0:
            self.group_counts[str(group_id)] = count - 1
            await self._save_group_counts()

    async def _load_user_checkin_data(self):
        if not self.user_checkin_file.exists():
            self.user_checkin_data = {}
            return
        try:
            content = await asyncio.to_thread(self.user_checkin_file.read_text, "utf-8")
            self.user_checkin_data = json.loads(content)
        except:
            self.user_checkin_data = {}

    async def _save_user_checkin_data(self):
        try:
            data = json.dumps(self.user_checkin_data, indent=4)
            await asyncio.to_thread(self.user_checkin_file.write_text, data, "utf-8")
        except:
            pass

    @filter.command("手办化签到", prefix_optional=True)
    async def on_checkin(self, event: AstrMessageEvent):
        if not self.conf.get("enable_checkin", False):
            yield event.plain_result("📅 签到未开启。")
            return
        
        uid = event.get_sender_id()
        today = datetime.now().strftime("%Y-%m-%d")
        
        if self.user_checkin_data.get(uid) == today:
            yield event.plain_result(f"已签到。剩余: {self._get_user_count(uid)}")
            return
        
        reward = int(self.conf.get("checkin_fixed_reward", 3))
        if self.conf.get("enable_random_checkin", False):
            max_r = int(self.conf.get("checkin_random_reward_max", 5))
            reward = random.randint(1, max(1, max_r))
            
        self.user_counts[uid] = self._get_user_count(uid) + reward
        await self._save_user_counts()
        self.user_checkin_data[uid] = today
        await self._save_user_checkin_data()
        
        yield event.plain_result(f"🎉 签到成功 +{reward}次。")

    @filter.command("手办化增加用户次数", prefix_optional=True)
    async def on_add_user_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
            
        text = event.message_str.strip()
        at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
        target, count = None, 0
        
        if at_seg:
            target = str(at_seg.qq)
            match = re.search(r"(\d+)\s*$", text)
            if match:
                count = int(match.group(1))
        else:
            match = re.search(r"(\d+)\s+(\d+)", text)
            if match:
                target, count = match.group(1), int(match.group(2))
            
        if target:
            self.user_counts[str(target)] = self._get_user_count(target) + count
            await self._save_user_counts()
            yield event.plain_result(f"✅ 已为 {target} 增加 {count} 次。")

    @filter.command("手办化增加群组次数", prefix_optional=True)
    async def on_add_group_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
            
        match = re.search(r"(\d+)\s+(\d+)", event.message_str.strip())
        if match:
            gid, count = match.group(1), int(match.group(2))
            self.group_counts[str(gid)] = self._get_group_count(gid) + count
            await self._save_group_counts()
            yield event.plain_result(f"✅ 已为群 {gid} 增加 {count} 次。")

    @filter.command("手办化查询次数", prefix_optional=True)
    async def on_query_counts(self, event: AstrMessageEvent):
        uid = event.get_sender_id()
        if self.is_global_admin(event):
            at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
            if at_seg:
                uid = str(at_seg.qq)
            else:
                match = re.search(r"(\d+)", event.message_str)
                if match:
                    uid = match.group(1)
        
        msg = f"👤 用户 {uid} 剩余: {self._get_user_count(uid)}"
        if gid := event.get_group_id():
            msg += f"\n👥 本群剩余: {self._get_group_count(gid)}"
        
        yield event.plain_result(msg)

    @filter.command("手办化添加key", prefix_optional=True)
    async def on_add_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
            
        new_keys = event.message_str.strip().split()[1:]
        if not new_keys:
            yield event.plain_result("格式错误。")
            return
        
        keys = self.conf.get("api_keys", [])
        added = [k for k in new_keys if k not in keys]
        keys.extend(added)
        self.conf["api_keys"] = keys
        if hasattr(self.conf, "save"):
            self.conf.save()
            
        yield event.plain_result(f"✅ 添加 {len(added)} 个Key。")

    @filter.command("手办化key列表", prefix_optional=True)
    async def on_list_keys(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
            
        keys = self.conf.get("api_keys", [])
        msg = "\n".join([f"{i+1}. {k[:8]}..." for i, k in enumerate(keys)])
        yield event.plain_result(f"🔑 通用 Key 池:\n{msg}")

    @filter.command("手办化删除key", prefix_optional=True)
    async def on_delete_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event):
            return
            
        parts = event.message_str.strip().split()
        if len(parts) < 2:
            yield event.plain_result("格式: #手办化删除key <序号|all>")
            return
        
        param = parts[1]
        keys = self.conf.get("api_keys", [])
        
        if param == "all":
            self.conf["api_keys"] = []
        elif param.isdigit():
            idx = int(param) - 1
            if 0 <= idx < len(keys):
                keys.pop(idx)
                self.conf["api_keys"] = keys
        
        if hasattr(self.conf, "save"):
            self.conf.save()
            
        yield event.plain_result("✅ 操作完成。")

    async def _get_pool_api_key(self) -> str | None:
        keys = self.conf.get("api_keys", [])
        if not keys:
            return None
            
        async with self.key_lock:
            key = keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(keys)
            return key

    def _find_url_recursively(self, data: Any) -> str | None:
        if isinstance(data, str):
            if data.startswith("http") and "://" in data:
                return data
            if "![image](" in data:
                match = re.search(r'!\[.*?\]\((http.*?)\)', data)
                if match:
                    return match.group(1)
            return None
            
        if isinstance(data, list):
            for item in data:
                res = self._find_url_recursively(item)
                if res:
                    return res
                    
        if isinstance(data, dict):
            if "url" in data and isinstance(data["url"], str) and data["url"].startswith("http"):
                return data["url"]
            if "image_url" in data:
                if isinstance(data["image_url"], str):
                    return data["image_url"]
                if isinstance(data["image_url"], dict) and "url" in data["image_url"]:
                    return data["image_url"]["url"]
            
            for v in data.values():
                res = self._find_url_recursively(v)
                if res:
                    return res
        return None

    def _extract_image_url_from_response(self, data: Dict[str, Any]) -> str | None:
        try:
            return data["choices"][0]["message"]["images"][0]["image_url"]["url"]
        except:
            pass
            
        try:
            return data["choices"][0]["message"]["images"][0]["url"]
        except:
            pass
            
        try:
            content = data["choices"][0]["message"]["content"]
            match = re.search(r'https?://[^\s<>")\]]+', content)
            if match:
                return match.group(0).rstrip(")>,'\"")
        except:
            pass
            
        try:
            if "data" in data and isinstance(data["data"], list):
                return data["data"][0]["url"]
        except:
            pass
            
        try:
            if "candidates" in data:
                parts = data["candidates"][0]["content"]["parts"]
                for p in parts:
                    if "text" in p:
                        match = re.search(r'https?://[^\s<>")\]]+', p["text"])
                        if match:
                            return match.group(0).rstrip(")>,'\"")
                    if "inlineData" in p:
                        return f"data:{p['inlineData']['mimeType']};base64,{p['inlineData']['data']}"
        except:
            pass
            
        return self._find_url_recursively(data)

    async def _call_api(self, image_bytes_list: List[bytes], prompt: str, override_model: str | None = None) -> bytes | str:
        api_url = self.conf.get("api_url")
        if not api_url:
            return "API URL 未配置"
            
        model_name = override_model or self.conf.get("model", "nano-banana")
        
        api_key = None
        c1 = self.conf.get("custom_model_1", "").strip()
        c2 = self.conf.get("custom_model_2", "").strip()
        
        if c1 and model_name == c1:
            api_key = self.conf.get("custom_model_1_key") or await self._get_pool_api_key()
        elif c2 and model_name == c2:
            api_key = self.conf.get("custom_model_2_key") or await self._get_pool_api_key()
        else:
            api_key = await self._get_pool_api_key()
            
        if not api_key:
            return "无可用 API Key"

        headers = {
            "Content-Type": "application/json",
            "Connection": "close"
        }
        
        api_mode = self.conf.get("api_mode", "generic")
        payload = {}
        final_url = api_url

        if api_mode == "gemini_official":
            if "models/" in api_url:
                base = api_url.split("models/")[0]
                final_url = f"{base}models/{model_name}:generateContent"
            else:
                base = api_url.rstrip("/")
                final_url = f"{base}/v1beta/models/{model_name}:generateContent"
            
            joiner = "&" if "?" in final_url else "?"
            if "goog" in api_url or "generativelanguage" in api_url:
                final_url += f"{joiner}key={api_key}"
            else:
                headers["Authorization"] = f"Bearer {api_key}"

            parts = [{"text": prompt}]
            for img in image_bytes_list:
                b64 = base64.b64encode(img).decode("utf-8")
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": b64
                    }
                })
            
            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {"maxOutputTokens": 1500}
            }
        
        else:
            headers["Authorization"] = f"Bearer {api_key}"
            
            content = [{"type": "text", "text": prompt}]
            for img in image_bytes_list:
                b64 = base64.b64encode(img).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })
            
            payload = {
                "model": model_name,
                "max_tokens": 1500,
                "stream": False,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content}
                ]
            }

        try:
            if not self.iwf:
                return "工作流未初始化"
                
            async with aiohttp.ClientSession() as session:
                async with session.post(final_url, json=payload, headers=headers, proxy=self.iwf.proxy, timeout=120) as resp:
                    if resp.status == 404 and api_mode == "gemini_official":
                        return f"API 404: 模型 '{model_name}' 可能不存在或 URL 配置错误。"
                    
                    if resp.status != 200:
                        text = await resp.text()
                        return f"API Error {resp.status}: {text[:200]}"
                    
                    data = await resp.json()
                    if "error" in data:
                        return json.dumps(data["error"])
                    
                    if "promptFeedback" in data:
                        pf = data["promptFeedback"]
                        if pf.get("blockReason"):
                            return f"Gemini 安全拦截: {pf['blockReason']}"
                    
                    url_or_b64 = self._extract_image_url_from_response(data)
                    
                    if not url_or_b64:
                        return f"生成失败，无图片数据。响应: {json.dumps(data)[:200]}..."
                    
                    if url_or_b64.startswith("data:"):
                        b64 = url_or_b64.split(",")[-1]
                        return base64.b64decode(b64)
                    else:
                        return await self.iwf._download_image(url_or_b64) or "下载图片失败"

        except asyncio.TimeoutError:
            return "请求超时"
        except Exception as e:
            logger.error(f"API Exception: {e}", exc_info=True)
            return f"系统错误: {e}"

    async def terminate(self):
        if self.iwf:
            await self.iwf.terminate()
        logger.info("[FigurinePro] 插件已终止")
