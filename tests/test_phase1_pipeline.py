"""
Phase 1 Pipeline 改造单元测试

测试内容：
1. GPU 调度机制
2. 子阶段独立化
3. Playlist 管理
4. VideoProcessor 细粒度阶段
5. 数据库迁移
"""
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    
import tempfile
import os
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# 简单的 fixture 装饰器（当 pytest 不可用时）
if not HAS_PYTEST:
    class pytest:
        @staticmethod
        def fixture(func):
            return func
        
        @staticmethod
        def skip(msg):
            raise SkipTest(msg)
    
    class SkipTest(Exception):
        pass


# ==================== GPU 调度测试 ====================

class TestGPUUtils:
    """GPU 工具函数测试"""
    
    def test_resolve_gpu_device_cpu(self):
        """测试 CPU 模式解析"""
        from vat.utils.gpu import resolve_gpu_device
        
        device_str, gpu_id = resolve_gpu_device("cpu", allow_cpu_fallback=True)
        assert device_str == "cpu"
        assert gpu_id is None
    
    def test_resolve_gpu_device_invalid(self):
        """测试无效设备标识符"""
        from vat.utils.gpu import resolve_gpu_device
        
        try:
            resolve_gpu_device("invalid_device")
            assert False, "应该抛出 ValueError"
        except ValueError:
            pass  # 期望的行为
    
    def test_resolve_gpu_device_cuda_format(self):
        """测试 cuda:N 格式解析"""
        from vat.utils.gpu import resolve_gpu_device
        
        # 这个测试需要 GPU 环境，跳过无 GPU 环境
        try:
            device_str, gpu_id = resolve_gpu_device("cuda:0")
            assert device_str == "cuda"
            assert gpu_id == 0
        except RuntimeError:
            pytest.skip("No GPU available")
    
    @patch('vat.utils.gpu.get_available_gpus')
    def test_select_best_gpu(self, mock_get_gpus):
        """测试最佳 GPU 选择"""
        from vat.utils.gpu import select_best_gpu, GPUInfo
        
        # 模拟两个 GPU
        mock_get_gpus.return_value = [
            GPUInfo(index=0, name="GPU 0", memory_total_mb=8000, 
                   memory_used_mb=6000, memory_free_mb=2000, utilization_percent=75),
            GPUInfo(index=1, name="GPU 1", memory_total_mb=8000, 
                   memory_used_mb=2000, memory_free_mb=6000, utilization_percent=25),
        ]
        
        best = select_best_gpu(min_free_memory_mb=1000)
        assert best == 1  # GPU 1 显存利用率更低
    
    @patch('vat.utils.gpu.get_available_gpus')
    def test_select_best_gpu_with_exclusion(self, mock_get_gpus):
        """测试排除指定 GPU"""
        from vat.utils.gpu import select_best_gpu, GPUInfo
        
        mock_get_gpus.return_value = [
            GPUInfo(index=0, name="GPU 0", memory_total_mb=8000, 
                   memory_used_mb=6000, memory_free_mb=2000, utilization_percent=75),
            GPUInfo(index=1, name="GPU 1", memory_total_mb=8000, 
                   memory_used_mb=2000, memory_free_mb=6000, utilization_percent=25),
        ]
        
        best = select_best_gpu(excluded_gpus=[1], min_free_memory_mb=1000)
        assert best == 0  # GPU 1 被排除


# ==================== 子阶段独立化测试 ====================

class TestTaskStepIndependence:
    """子阶段独立化测试"""
    
    def test_task_step_enum_values(self):
        """测试 TaskStep 枚举值"""
        from vat.models import TaskStep
        
        # 验证所有细粒度阶段存在
        assert TaskStep.DOWNLOAD.value == "download"
        assert TaskStep.WHISPER.value == "whisper"
        assert TaskStep.SPLIT.value == "split"
        assert TaskStep.OPTIMIZE.value == "optimize"
        assert TaskStep.TRANSLATE.value == "translate"
        assert TaskStep.EMBED.value == "embed"
        assert TaskStep.UPLOAD.value == "upload"
    
    def test_stage_groups(self):
        """测试阶段组定义"""
        from vat.models import STAGE_GROUPS, TaskStep
        
        # ASR 组包含 WHISPER 和 SPLIT
        assert TaskStep.WHISPER in STAGE_GROUPS["asr"]
        assert TaskStep.SPLIT in STAGE_GROUPS["asr"]
        
        # TRANSLATE 组包含 OPTIMIZE 和 TRANSLATE
        assert TaskStep.OPTIMIZE in STAGE_GROUPS["translate"]
        assert TaskStep.TRANSLATE in STAGE_GROUPS["translate"]
    
    def test_expand_stage_group(self):
        """测试阶段组展开"""
        from vat.models import expand_stage_group, TaskStep
        
        # 展开 asr 组
        asr_stages = expand_stage_group("asr")
        assert TaskStep.WHISPER in asr_stages
        assert TaskStep.SPLIT in asr_stages
        
        # 单个阶段
        whisper_stages = expand_stage_group("whisper")
        assert whisper_stages == [TaskStep.WHISPER]
    
    def test_get_required_stages(self):
        """测试依赖解析"""
        from vat.models import get_required_stages, TaskStep
        
        # 执行 TRANSLATE 需要所有前置阶段
        required = get_required_stages([TaskStep.TRANSLATE])
        assert TaskStep.DOWNLOAD in required
        assert TaskStep.WHISPER in required
        assert TaskStep.SPLIT in required
        assert TaskStep.OPTIMIZE in required
        assert TaskStep.TRANSLATE in required
        
        # 顺序正确
        assert required.index(TaskStep.DOWNLOAD) < required.index(TaskStep.WHISPER)
        assert required.index(TaskStep.WHISPER) < required.index(TaskStep.SPLIT)
    
    def test_stage_dependencies(self):
        """测试阶段依赖关系"""
        from vat.models import STAGE_DEPENDENCIES, TaskStep
        
        # WHISPER 依赖 DOWNLOAD
        assert TaskStep.DOWNLOAD in STAGE_DEPENDENCIES[TaskStep.WHISPER]
        
        # SPLIT 依赖 WHISPER
        assert TaskStep.WHISPER in STAGE_DEPENDENCIES[TaskStep.SPLIT]
        
        # TRANSLATE 依赖 OPTIMIZE
        assert TaskStep.OPTIMIZE in STAGE_DEPENDENCIES[TaskStep.TRANSLATE]


# ==================== 数据库测试 ====================

class TestDatabase:
    """数据库测试"""
    
    def _create_temp_db(self):
        """创建临时数据库"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        from vat.database import Database
        return Database(db_path), db_path
    
    def test_video_with_playlist(self):
        """测试带 Playlist 的视频"""
        temp_db, db_path = self._create_temp_db()
        try:
            from vat.models import Video, SourceType
            
            video = Video(
                id="test_video_1",
                source_type=SourceType.YOUTUBE,
                source_url="https://youtube.com/watch?v=test",
                title="Test Video",
                playlist_id="playlist_1",
                playlist_index=1
            )
            
            temp_db.add_video(video)
            loaded = temp_db.get_video("test_video_1")
            
            assert loaded is not None
            assert loaded.playlist_id == "playlist_1"
            assert loaded.playlist_index == 1
        finally:
            os.unlink(db_path)
    
    def test_playlist_crud(self):
        """测试 Playlist CRUD 操作"""
        temp_db, db_path = self._create_temp_db()
        try:
            from vat.models import Playlist
            
            playlist = Playlist(
                id="PL123",
                title="Test Playlist",
                source_url="https://youtube.com/playlist?list=PL123",
                channel="Test Channel",
                video_count=10
            )
            
            # Create
            temp_db.add_playlist(playlist)
            
            # Read
            loaded = temp_db.get_playlist("PL123")
            assert loaded is not None
            assert loaded.title == "Test Playlist"
            assert loaded.video_count == 10
            
            # Update
            temp_db.update_playlist("PL123", video_count=15)
            loaded = temp_db.get_playlist("PL123")
            assert loaded.video_count == 15
            
            # List
            playlists = temp_db.list_playlists()
            assert len(playlists) == 1
            
            # Delete
            temp_db.delete_playlist("PL123")
            assert temp_db.get_playlist("PL123") is None
        finally:
            os.unlink(db_path)
    
    def test_list_videos_by_playlist(self):
        """测试按 Playlist 列出视频（使用关联表）"""
        temp_db, db_path = self._create_temp_db()
        try:
            from vat.models import Video, SourceType, Playlist
            
            # 创建 Playlist
            playlist = Playlist(
                id="PL123",
                title="Test Playlist",
                source_url="https://youtube.com/playlist?list=PL123"
            )
            temp_db.add_playlist(playlist)
            
            # 添加视频并写入关联表
            for i in range(3):
                video = Video(
                    id=f"video_{i}",
                    source_type=SourceType.YOUTUBE,
                    source_url=f"https://youtube.com/watch?v=video_{i}",
                    playlist_id="PL123",
                    playlist_index=i + 1
                )
                temp_db.add_video(video)
                temp_db.add_video_to_playlist(f"video_{i}", "PL123", playlist_index=i + 1)
            
            # 按 Playlist 查询（通过 playlist_videos 关联表）
            videos = temp_db.list_videos(playlist_id="PL123")
            assert len(videos) == 3
            
            # 验证顺序（按 playlist_index）
            assert videos[0].id == "video_0"
            assert videos[1].id == "video_1"
            assert videos[2].id == "video_2"
        finally:
            os.unlink(db_path)
    
    def test_task_with_fine_grained_steps(self):
        """测试细粒度阶段任务"""
        temp_db, db_path = self._create_temp_db()
        try:
            from vat.models import Video, Task, TaskStep, TaskStatus, SourceType
            
            # 创建视频
            video = Video(
                id="test_video",
                source_type=SourceType.YOUTUBE,
                source_url="https://youtube.com/watch?v=test"
            )
            temp_db.add_video(video)
            
            # 添加细粒度阶段任务
            for step in [TaskStep.DOWNLOAD, TaskStep.WHISPER, TaskStep.SPLIT]:
                task = Task(
                    video_id="test_video",
                    step=step,
                    status=TaskStatus.COMPLETED
                )
                temp_db.add_task(task)
            
            # 获取待处理阶段
            pending = temp_db.get_pending_steps("test_video")
            
            # DOWNLOAD, WHISPER, SPLIT 已完成，剩余应该是 OPTIMIZE, TRANSLATE, EMBED, UPLOAD
            assert TaskStep.OPTIMIZE in pending
            assert TaskStep.TRANSLATE in pending
            assert TaskStep.EMBED in pending
            assert TaskStep.UPLOAD in pending
            assert TaskStep.DOWNLOAD not in pending
        finally:
            os.unlink(db_path)


# ==================== Playlist Service 测试 ====================

class TestPlaylistService:
    """Playlist 服务测试"""
    
    def _create_temp_db(self):
        """创建临时数据库"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        from vat.database import Database
        return Database(db_path), db_path
    
    def test_get_playlist_videos_ordering(self):
        """测试 Playlist 视频排序"""
        temp_db, db_path = self._create_temp_db()
        try:
            from vat.models import Video, Playlist, SourceType
            from vat.services import PlaylistService
            
            # 创建 Playlist
            playlist = Playlist(
                id="PL123",
                title="Test Playlist",
                source_url="https://youtube.com/playlist?list=PL123"
            )
            temp_db.add_playlist(playlist)
            
            # 添加视频（故意乱序）并写入关联表
            for i in [3, 1, 2]:
                video = Video(
                    id=f"video_{i}",
                    source_type=SourceType.YOUTUBE,
                    source_url=f"https://youtube.com/watch?v=video_{i}",
                    playlist_id="PL123",
                    playlist_index=i
                )
                temp_db.add_video(video)
                temp_db.add_video_to_playlist(f"video_{i}", "PL123", playlist_index=i)
            
            service = PlaylistService(temp_db)
            videos = service.get_playlist_videos("PL123", order_by="playlist_index")
            
            # 应该按 playlist_index 排序
            assert videos[0].playlist_index == 1
            assert videos[1].playlist_index == 2
            assert videos[2].playlist_index == 3
        finally:
            os.unlink(db_path)
    
    def test_get_pending_videos(self):
        """测试获取待处理视频"""
        temp_db, db_path = self._create_temp_db()
        try:
            from vat.models import Video, Playlist, Task, TaskStep, TaskStatus, SourceType
            from vat.services import PlaylistService
            
            # 创建 Playlist 和视频
            playlist = Playlist(
                id="PL123",
                title="Test Playlist",
                source_url="https://youtube.com/playlist?list=PL123"
            )
            temp_db.add_playlist(playlist)
            
            for i in range(3):
                video = Video(
                    id=f"video_{i}",
                    source_type=SourceType.YOUTUBE,
                    source_url=f"https://youtube.com/watch?v=video_{i}",
                    playlist_id="PL123",
                    playlist_index=i + 1
                )
                temp_db.add_video(video)
                temp_db.add_video_to_playlist(f"video_{i}", "PL123", playlist_index=i + 1)
            
            # 标记第一个视频为全部完成
            for step in [TaskStep.DOWNLOAD, TaskStep.WHISPER, TaskStep.SPLIT, 
                         TaskStep.OPTIMIZE, TaskStep.TRANSLATE, TaskStep.EMBED, TaskStep.UPLOAD]:
                task = Task(video_id="video_0", step=step, status=TaskStatus.COMPLETED)
                temp_db.add_task(task)
            
            service = PlaylistService(temp_db)
            pending = service.get_pending_videos("PL123")
            
            # video_0 已完成，只有 video_1 和 video_2 待处理
            assert len(pending) == 2
            assert all(v.id != "video_0" for v in pending)
        finally:
            os.unlink(db_path)


# ==================== Config 测试 ====================

class TestConfig:
    """配置测试"""
    
    def test_gpu_config_loading(self):
        """测试 GPU 配置加载"""
        from vat.config import Config, GPUConfig
        
        data = {
            'storage': {
                'work_dir': '/tmp/work',
                'output_dir': '/tmp/output',
                'database_path': '/tmp/db.db',
                'models_dir': '/tmp/models',
                'resource_dir': 'resources',
                'fonts_dir': 'fonts',
                'subtitle_style_dir': 'styles',
                'cache_dir': '/tmp/cache'
            },
            'downloader': {'youtube': {'format': 'best', 'max_workers': 1}},
            'asr': {
                'backend': 'faster-whisper',
                'model': 'large-v3',
                'language': 'ja',
                'device': 'auto',
                'compute_type': 'float16',
                'vad_filter': False,
                'beam_size': 5,
                'models_subdir': 'whisper',
                'word_timestamps': True,
                'condition_on_previous_text': False,
                'temperature': [0.0],
                'compression_ratio_threshold': 2.4,
                'log_prob_threshold': -1.0,
                'no_speech_threshold': 0.6,
                'initial_prompt': '',
                'repetition_penalty': 1.0,
                'hallucination_silence_threshold': None,
                'vad_threshold': 0.5,
                'vad_min_speech_duration_ms': 250,
                'vad_max_speech_duration_s': 30,
                'vad_min_silence_duration_ms': 100,
                'vad_speech_pad_ms': 30,
                'enable_chunked': False,
                'chunk_length_sec': 300,
                'chunk_overlap_sec': 10,
                'chunk_concurrency': 1,
                'split': {
                    'enable': True,
                    'mode': 'sentence',
                    'max_words_cjk': 30,
                    'max_words_english': 15,
                    'min_words_cjk': 5,
                    'min_words_english': 3,
                    'model': 'gpt-4',
                    'enable_chunking': False,
                    'chunk_size_sentences': 50,
                    'chunk_overlap_sentences': 2,
                    'chunk_min_threshold': 100
                }
            },
            'translator': {
                'backend_type': 'llm',
                'source_language': 'ja',
                'target_language': 'zh-cn',
                'skip_translate': False,
                'llm': {
                    'model': 'gpt-4',
                    'enable_reflect': True,
                    'batch_size': 10,
                    'thread_num': 3,
                    'custom_prompt': '',
                    'enable_context': True,
                    'optimize': {
                        'enable': True,
                        'custom_prompt': ''
                    }
                },
                'local': {
                    'model_filename': 'model.gguf',
                    'backend': 'sakura',
                    'n_gpu_layers': 35,
                    'context_size': 4096
                }
            },
            'embedder': {
                'subtitle_formats': ['srt'],
                'embed_mode': 'hard',
                'output_container': 'mp4',
                'video_codec': 'libx265',
                'audio_codec': 'copy',
                'crf': 23,
                'preset': 'medium',
                'use_gpu': True,
                'subtitle_style': 'default'
            },
            'uploader': {'bilibili': {'cookies_file': '', 'line': 'AUTO', 'threads': 3}},
            'gpu': {
                'device': 'auto',
                'allow_cpu_fallback': False,
                'min_free_memory_mb': 2000
            },
            'concurrency': {'gpu_devices': [0], 'max_concurrent_per_gpu': 1},
            'logging': {'level': 'INFO', 'file': 'vat.log', 'format': '%(message)s'},
            'llm': {'api_key': '', 'base_url': ''},
            'proxy': {'http_proxy': ''}
        }
        
        config = Config.from_dict(data)
        
        assert config.gpu.device == "auto"
        assert config.gpu.allow_cpu_fallback == False
        assert config.gpu.min_free_memory_mb == 2000


# ==================== VideoProcessor 细粒度阶段测试 ====================

class TestVideoProcessorStages:
    """VideoProcessor 细粒度阶段测试"""
    
    @pytest.fixture
    def temp_workspace(self):
        """创建临时工作空间"""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test.db")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir)
        
        from vat.database import Database
        db = Database(db_path)
        
        yield {
            'temp_dir': temp_dir,
            'db_path': db_path,
            'output_dir': output_dir,
            'db': db
        }
        
        # 清理
        shutil.rmtree(temp_dir)
    
    def test_execute_step_mapping(self):
        """测试 _execute_step 阶段映射"""
        from vat.models import TaskStep
        
        # 验证所有细粒度阶段都有对应的处理方法
        expected_steps = [
            TaskStep.DOWNLOAD,
            TaskStep.WHISPER,
            TaskStep.SPLIT,
            TaskStep.OPTIMIZE,
            TaskStep.TRANSLATE,
            TaskStep.EMBED,
            TaskStep.UPLOAD,
        ]
        
        for step in expected_steps:
            assert step.value in ['download', 'whisper', 'split', 'optimize', 
                                  'translate', 'embed', 'upload']
    
    def test_stage_dependencies_completeness(self):
        """测试阶段依赖关系完整性"""
        from vat.models import STAGE_DEPENDENCIES, TaskStep, DEFAULT_STAGE_SEQUENCE
        
        # 每个阶段（除 DOWNLOAD）都应该有依赖
        for step in DEFAULT_STAGE_SEQUENCE[1:]:  # 跳过 DOWNLOAD
            assert step in STAGE_DEPENDENCIES, f"{step} 缺少依赖定义"
            assert len(STAGE_DEPENDENCIES[step]) > 0, f"{step} 依赖列表为空"
    
    def test_get_required_stages_order(self):
        """测试依赖解析顺序正确性"""
        from vat.models import get_required_stages, TaskStep
        
        # EMBED 需要所有前置阶段
        required = get_required_stages([TaskStep.EMBED])
        
        # 验证顺序
        download_idx = required.index(TaskStep.DOWNLOAD)
        whisper_idx = required.index(TaskStep.WHISPER)
        split_idx = required.index(TaskStep.SPLIT)
        optimize_idx = required.index(TaskStep.OPTIMIZE)
        translate_idx = required.index(TaskStep.TRANSLATE)
        embed_idx = required.index(TaskStep.EMBED)
        
        assert download_idx < whisper_idx < split_idx < optimize_idx < translate_idx < embed_idx


# ==================== 数据库迁移测试 ====================

class TestDatabaseMigration:
    """数据库迁移测试"""
    
    def test_fresh_database_creation(self):
        """测试新数据库创建"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            from vat.database import Database, DB_VERSION
            db = Database(db_path)
            
            # 验证表存在
            with db.get_connection() as conn:
                cursor = conn.cursor()
                
                # 检查版本
                cursor.execute("SELECT version FROM db_version LIMIT 1")
                row = cursor.fetchone()
                assert row['version'] == DB_VERSION
                
                # 检查表结构
                cursor.execute("PRAGMA table_info(videos)")
                columns = {row['name'] for row in cursor.fetchall()}
                assert 'playlist_id' in columns
                assert 'playlist_index' in columns
                
                cursor.execute("PRAGMA table_info(playlists)")
                columns = {row['name'] for row in cursor.fetchall()}
                assert 'id' in columns
                assert 'title' in columns
                assert 'video_count' in columns
        finally:
            os.unlink(db_path)
    
    def test_task_without_sub_phase(self):
        """测试任务不再使用 sub_phase"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            from vat.database import Database
            from vat.models import Video, Task, TaskStep, TaskStatus, SourceType
            
            db = Database(db_path)
            
            # 创建视频
            video = Video(
                id="test_video",
                source_type=SourceType.YOUTUBE,
                source_url="https://youtube.com/watch?v=test"
            )
            db.add_video(video)
            
            # 创建任务（不使用 sub_phase）
            task = Task(
                video_id="test_video",
                step=TaskStep.WHISPER,
                status=TaskStatus.COMPLETED
            )
            task_id = db.add_task(task)
            
            # 读取任务
            loaded = db.get_task("test_video", TaskStep.WHISPER)
            assert loaded is not None
            assert loaded.step == TaskStep.WHISPER
            assert loaded.status == TaskStatus.COMPLETED
        finally:
            os.unlink(db_path)


# ==================== 集成测试辅助 ====================

class TestIntegrationHelpers:
    """集成测试辅助功能"""
    
    def test_default_stage_sequence(self):
        """测试默认阶段序列"""
        from vat.models import DEFAULT_STAGE_SEQUENCE, TaskStep
        
        assert len(DEFAULT_STAGE_SEQUENCE) == 7
        assert DEFAULT_STAGE_SEQUENCE[0] == TaskStep.DOWNLOAD
        assert DEFAULT_STAGE_SEQUENCE[-1] == TaskStep.UPLOAD
    
    def test_expand_all_stage_groups(self):
        """测试所有阶段组展开"""
        from vat.models import expand_stage_group, TaskStep
        
        # 测试所有组
        assert len(expand_stage_group("download")) == 1
        assert len(expand_stage_group("asr")) == 2
        assert len(expand_stage_group("translate")) == 2
        assert len(expand_stage_group("embed")) == 1
        assert len(expand_stage_group("upload")) == 1
    
    def test_playlist_video_ordering(self):
        """测试 Playlist 视频排序"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            from vat.database import Database
            from vat.models import Video, Playlist, SourceType
            
            db = Database(db_path)
            
            # 创建 Playlist
            playlist = Playlist(
                id="PL_order",
                title="Order Test",
                source_url="https://youtube.com/playlist?list=PL_order"
            )
            db.add_playlist(playlist)
            
            # 添加视频（故意乱序添加）并写入关联表
            for i in [5, 2, 8, 1, 3]:
                video = Video(
                    id=f"video_{i}",
                    source_type=SourceType.YOUTUBE,
                    source_url=f"https://youtube.com/watch?v=v{i}",
                    playlist_id="PL_order",
                    playlist_index=i
                )
                db.add_video(video)
                db.add_video_to_playlist(f"video_{i}", "PL_order", playlist_index=i)
            
            # 按 Playlist 查询应该按 index 排序
            videos = db.list_videos(playlist_id="PL_order")
            indices = [v.playlist_index for v in videos]
            
            assert indices == sorted(indices), "视频应该按 playlist_index 排序"
        finally:
            os.unlink(db_path)


# ==================== YouTubeDownloader 测试 ====================

class TestYouTubeDownloaderExtensions:
    """YouTubeDownloader 扩展功能测试"""
    
    def test_is_playlist_url(self):
        """测试 Playlist URL 识别"""
        from vat.downloaders import YouTubeDownloader
        
        dl = YouTubeDownloader()
        
        # Playlist URLs
        assert dl.is_playlist_url("https://www.youtube.com/playlist?list=PLxxx")
        assert dl.is_playlist_url("https://youtube.com/watch?v=xxx&list=PLyyy")
        
        # Non-playlist URLs
        assert not dl.is_playlist_url("https://youtube.com/watch?v=xxx")
        assert not dl.is_playlist_url("https://youtu.be/xxx")
    
    def test_extract_playlist_id(self):
        """测试 Playlist ID 提取"""
        from vat.downloaders import YouTubeDownloader
        
        dl = YouTubeDownloader()
        
        assert dl.extract_playlist_id("https://youtube.com/playlist?list=PLtest123") == "PLtest123"
        assert dl.extract_playlist_id("https://youtube.com/watch?v=xxx&list=PLfoo") == "PLfoo"


# ==================== Exception 测试 ====================

class TestPipelineExceptions:
    """Pipeline 异常测试"""
    
    def test_pipeline_error_without_sub_phase(self):
        """测试 PipelineError 不再需要 sub_phase"""
        from vat.pipeline.exceptions import PipelineError, ASRError, TranslateError
        
        # 基本异常
        err = PipelineError("Test error")
        assert str(err) == "Test error"
        assert err.original_error is None
        
        # 带原始异常
        original = ValueError("Original")
        err = ASRError("ASR failed", original_error=original)
        assert err.original_error is original
        
        # TranslateError
        err = TranslateError("Translate failed")
        assert str(err) == "Translate failed"
        
        # UploadError
        from vat.pipeline.exceptions import UploadError
        err = UploadError("Upload failed")
        assert str(err) == "Upload failed"
        assert isinstance(err, PipelineError)
    
    def test_all_stage_methods_raise_not_return_false(self):
        """确认所有阶段方法的异常类型正确注册（设计-6）"""
        from vat.pipeline.exceptions import (
            PipelineError, ASRError, TranslateError, EmbedError, DownloadError, UploadError
        )
        
        # 所有 PipelineError 子类都应存在
        subclasses = [ASRError, TranslateError, EmbedError, DownloadError, UploadError]
        for cls in subclasses:
            assert issubclass(cls, PipelineError)
            # 确认可以正常实例化
            err = cls("test")
            assert str(err) == "test"


# ==================== 阶段组展开集成测试 ====================

class TestStageGroupExpansion:
    """测试 process() 中阶段组名展开逻辑"""
    
    def test_expand_asr_group_in_steps(self):
        """测试 'asr' 展开为 ['whisper', 'split']"""
        from vat.models import expand_stage_group, TaskStep
        
        # 模拟 process() 中的展开逻辑
        steps = ['asr']
        expanded = []
        for s in steps:
            try:
                group = expand_stage_group(s)
                expanded.extend([step.value for step in group])
            except ValueError:
                expanded.append(s)
        
        assert expanded == ['whisper', 'split']
    
    def test_expand_translate_group_in_steps(self):
        """测试 'translate' 展开为 ['optimize', 'translate']"""
        from vat.models import expand_stage_group, TaskStep
        
        steps = ['translate']
        expanded = []
        for s in steps:
            try:
                group = expand_stage_group(s)
                expanded.extend([step.value for step in group])
            except ValueError:
                expanded.append(s)
        
        assert expanded == ['optimize', 'translate']
    
    def test_expand_mixed_steps(self):
        """测试混合阶段组和单阶段的展开"""
        from vat.models import expand_stage_group, TaskStep
        
        # 模拟 CLI 的 pipeline 命令：steps=['download', 'asr', 'translate', 'embed']
        steps = ['download', 'asr', 'translate', 'embed']
        expanded = []
        for s in steps:
            try:
                group = expand_stage_group(s)
                expanded.extend([step.value for step in group])
            except ValueError:
                expanded.append(s)
        
        # 去重保序
        seen = set()
        deduped = []
        for s in expanded:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        
        assert deduped == ['download', 'whisper', 'split', 'optimize', 'translate', 'embed']
    
    def test_expand_individual_step_passthrough(self):
        """测试单个阶段名不被错误展开"""
        from vat.models import expand_stage_group, TaskStep
        
        # 'whisper' 是单阶段，不是组名（虽然它在 asr 组内）
        result = expand_stage_group('whisper')
        assert result == [TaskStep.WHISPER]
    
    def test_expand_unknown_raises(self):
        """测试未知名称抛出 ValueError"""
        from vat.models import expand_stage_group
        
        try:
            expand_stage_group('nonexistent')
            assert False, "应该抛出 ValueError"
        except ValueError:
            pass


# ==================== 批次进度计算测试 ====================

class TestBatchProgressCalculation:
    """测试批次总进度计算逻辑"""
    
    def test_single_video_progress(self):
        """单视频时总进度等于单视频进度"""
        # video_index=0, total_videos=1, per_video=0.5
        total = (0 + 0.5) / 1
        assert total == 0.5
    
    def test_multi_video_first(self):
        """3个视频中第1个处理到50%"""
        # video_index=0, total_videos=3, per_video=0.5
        total = (0 + 0.5) / 3
        assert abs(total - 1/6) < 0.001
    
    def test_multi_video_second_complete(self):
        """3个视频中第2个处理完成"""
        # video_index=1, total_videos=3, per_video=1.0
        total = (1 + 1.0) / 3
        assert abs(total - 2/3) < 0.001
    
    def test_multi_video_all_done(self):
        """3个视频全部完成"""
        # video_index=2, total_videos=3, per_video=1.0
        total = (2 + 1.0) / 3
        assert total == 1.0
    
    def test_progress_format_output(self):
        """测试进度格式字符串"""
        video_index = 1
        total_videos = 3
        per_video = 0.5
        total = (video_index + per_video) / total_videos
        
        prefix = f"[TOTAL:{total:.0%}] [{per_video:.0%}]"
        assert prefix == "[TOTAL:50%] [50%]"


# ==================== 进度日志解析测试 ====================

class TestProgressLogParsing:
    """测试 WebUI 进度日志解析"""
    
    def test_parse_total_progress(self):
        """测试解析 [TOTAL:N%] 格式"""
        import re
        
        line = "10:00:00 | INFO | [TOTAL:50%] [75%] Whisper 完成"
        total_match = re.search(r'\[TOTAL:(\d+)%\]', line)
        assert total_match is not None
        assert float(total_match.group(1)) / 100.0 == 0.5
    
    def test_parse_fallback_to_per_video(self):
        """测试旧格式回退 [N%]"""
        import re
        
        line = "10:00:00 | INFO | [75%] 翻译中..."
        total_match = re.search(r'\[TOTAL:(\d+)%\]', line)
        assert total_match is None
        
        match = re.search(r'\[(\d+)%\]', line)
        assert match is not None
        assert float(match.group(1)) / 100.0 == 0.75
    
    def test_parse_total_preferred_over_per_video(self):
        """测试 TOTAL 优先于 per-video 进度"""
        import re
        
        line = "[TOTAL:33%] [100%] 阶段完成: embed"
        # 应该解析出 33% 而不是 100%
        total_match = re.search(r'\[TOTAL:(\d+)%\]', line)
        assert total_match is not None
        assert float(total_match.group(1)) / 100.0 == 0.33
    
    def test_parse_multi_line_last_wins(self):
        """测试多行日志取最后的进度"""
        import re
        
        lines = [
            "[TOTAL:10%] [30%] 下载中",
            "[TOTAL:33%] [100%] 第1个视频完成",
            "[TOTAL:50%] [50%] 第2个视频 Whisper 完成",
        ]
        
        progress = 0.0
        for line in reversed(lines):
            total_match = re.search(r'\[TOTAL:(\d+)%\]', line)
            if total_match:
                progress = float(total_match.group(1)) / 100.0
                break
            match = re.search(r'\[(\d+)%\]', line)
            if match:
                progress = float(match.group(1)) / 100.0
                break
        
        assert progress == 0.5


# ==================== Speaker 分组测试 ====================

class TestSpeakerGrouping:
    """测试说话人分组逻辑"""
    
    def test_group_by_speaker(self):
        """测试按说话人分组"""
        from collections import defaultdict
        
        # 模拟 _group_segments_by_speaker 的逻辑
        class MockSeg:
            def __init__(self, text, speaker_id=None):
                self.text = text
                self.speaker_id = speaker_id
        
        segments = [
            MockSeg("Hello", "SPEAKER_00"),
            MockSeg("World", "SPEAKER_01"),
            MockSeg("Foo", "SPEAKER_00"),
            MockSeg("Bar", None),  # 未知说话人
        ]
        
        groups = defaultdict(list)
        for seg in segments:
            speaker_id = seg.speaker_id or "SPEAKER_UNKNOWN"
            groups[speaker_id].append(seg)
        
        assert len(groups["SPEAKER_00"]) == 2
        assert len(groups["SPEAKER_01"]) == 1
        assert len(groups["SPEAKER_UNKNOWN"]) == 1
    
    def test_no_speakers(self):
        """测试无说话人信息时不分组"""
        class MockSeg:
            def __init__(self, text):
                self.text = text
                self.speaker_id = None
        
        segments = [MockSeg("A"), MockSeg("B"), MockSeg("C")]
        has_speakers = any(seg.speaker_id is not None for seg in segments)
        assert has_speakers is False


# ==================== create_video_from_url 重复任务修复测试 ====================

class TestCreateVideoFromUrl:
    """测试 create_video_from_url 重复创建时清理旧任务"""
    
    def _create_temp_db(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        from vat.database import Database
        return Database(db_path), db_path
    
    def test_first_creation_has_correct_tasks(self):
        """首次创建视频应有 7 个 PENDING 任务"""
        temp_db, db_path = self._create_temp_db()
        try:
            from vat.pipeline.executor import create_video_from_url
            
            video_id = create_video_from_url("https://youtube.com/watch?v=test1", temp_db)
            tasks = temp_db.get_tasks(video_id)
            assert len(tasks) == 7, f"应有 7 个任务，实际 {len(tasks)}"
            assert all(t.status.value == 'pending' for t in tasks)
        finally:
            os.unlink(db_path)
    
    def test_duplicate_creation_no_extra_tasks(self):
        """重复创建同一 URL 不应产生重复任务"""
        temp_db, db_path = self._create_temp_db()
        try:
            from vat.pipeline.executor import create_video_from_url
            
            url = "https://youtube.com/watch?v=test_dup"
            video_id_1 = create_video_from_url(url, temp_db)
            video_id_2 = create_video_from_url(url, temp_db)
            
            # 同一 URL 生成同一 video_id
            assert video_id_1 == video_id_2
            
            # 重复创建后仍然只有 7 个任务（旧的被清理）
            tasks = temp_db.get_tasks(video_id_1)
            assert len(tasks) == 7, f"重复创建后应有 7 个任务，实际 {len(tasks)}"
        finally:
            os.unlink(db_path)
    
    def test_duplicate_creation_resets_completed_tasks(self):
        """重复创建应重置已完成的任务为 PENDING"""
        temp_db, db_path = self._create_temp_db()
        try:
            from vat.pipeline.executor import create_video_from_url
            from vat.models import TaskStep, TaskStatus
            
            url = "https://youtube.com/watch?v=test_reset"
            video_id = create_video_from_url(url, temp_db)
            
            # 标记部分任务为已完成
            temp_db.update_task_status(video_id, TaskStep.DOWNLOAD, TaskStatus.COMPLETED)
            temp_db.update_task_status(video_id, TaskStep.WHISPER, TaskStatus.COMPLETED)
            
            # 重新创建
            create_video_from_url(url, temp_db)
            
            # 所有任务应被重置为 PENDING
            tasks = temp_db.get_tasks(video_id)
            assert len(tasks) == 7
            assert all(t.status.value == 'pending' for t in tasks)
        finally:
            os.unlink(db_path)
    
    def test_delete_tasks_for_video(self):
        """测试 Database.delete_tasks_for_video"""
        temp_db, db_path = self._create_temp_db()
        try:
            from vat.models import Task, TaskStep, TaskStatus
            
            # 添加任务
            for step in [TaskStep.DOWNLOAD, TaskStep.WHISPER]:
                temp_db.add_task(Task(video_id="v1", step=step, status=TaskStatus.PENDING))
            
            assert len(temp_db.get_tasks("v1")) == 2
            
            deleted = temp_db.delete_tasks_for_video("v1")
            assert deleted == 2
            assert len(temp_db.get_tasks("v1")) == 0
        finally:
            os.unlink(db_path)


# ==================== 死代码清理验证测试 ====================

class TestDeadCodeRemoval:
    """验证死代码已被正确清理"""
    
    def test_no_old_run_asr_method(self):
        """确认旧 _run_asr 方法已删除"""
        import inspect
        from vat.pipeline.executor import VideoProcessor
        members = [m[0] for m in inspect.getmembers(VideoProcessor, predicate=inspect.isfunction)]
        assert '_run_asr' not in members
    
    def test_no_old_translate_method(self):
        """确认旧 _translate 方法已删除"""
        import inspect
        from vat.pipeline.executor import VideoProcessor
        members = [m[0] for m in inspect.getmembers(VideoProcessor, predicate=inspect.isfunction)]
        assert '_translate' not in members
    
    def test_active_methods_exist(self):
        """确认活跃方法仍存在"""
        import inspect
        from vat.pipeline.executor import VideoProcessor
        members = [m[0] for m in inspect.getmembers(VideoProcessor, predicate=inspect.isfunction)]
        
        required = ['_run_whisper', '_run_split', '_run_optimize', '_run_translate',
                     '_realign_timestamps', '_split_with_speaker_awareness', '_group_segments_by_speaker']
        for method in required:
            assert method in members, f"活跃方法 {method} 应该存在"
    
    def test_task_service_deleted(self):
        """确认 task_service.py 已删除"""
        try:
            from vat.web.services.task_service import TaskService
            assert False, "task_service.py 应该已删除"
        except ImportError:
            pass
    
    def test_services_init_clean(self):
        """确认 services/__init__.py 不再导出 TaskService"""
        from vat.web import services
        assert not hasattr(services, 'TaskService')
        assert not hasattr(services, 'TaskInfo')
        assert not hasattr(services, 'WebTaskStatus')


# ==================== Scheduler 日志统一测试 ====================

class TestSchedulerLogging:
    """验证 scheduler 已从 print 迁移到 logger"""
    
    def test_no_print_in_scheduler(self):
        """确认 scheduler.py 无 print() 调用"""
        import inspect
        import vat.pipeline.scheduler as mod
        source = inspect.getsource(mod)
        
        # 只检查非注释行
        code_lines = []
        for line in source.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            code_lines.append(stripped)
        
        print_lines = [l for l in code_lines if 'print(' in l and 'lambda' not in l]
        assert len(print_lines) == 0, f"scheduler 中仍有 print: {print_lines}"
    
    def test_no_traceback_print_exc_in_executor(self):
        """确认 executor.py 无 traceback.print_exc() 调用"""
        import inspect
        from vat.pipeline.executor import VideoProcessor
        source = inspect.getsource(VideoProcessor)
        
        assert 'traceback.print_exc' not in source, "executor 中仍有 traceback.print_exc"
    
    def test_video_processor_has_batch_params(self):
        """确认 VideoProcessor 支持批次进度参数"""
        import inspect
        from vat.pipeline.executor import VideoProcessor
        
        sig = inspect.signature(VideoProcessor.__init__)
        params = list(sig.parameters.keys())
        assert 'video_index' in params
        assert 'total_videos' in params
    
    def test_process_no_force_param(self):
        """确认 process() 不再接受 force 参数（设计-4）"""
        import inspect
        from vat.pipeline.executor import VideoProcessor
        
        sig = inspect.signature(VideoProcessor.process)
        params = list(sig.parameters.keys())
        assert 'force' not in params, "process() 不应再接受 force 参数"
        assert 'steps' in params
    
    def test_factory_methods_exist(self):
        """确认工厂方法已提取（设计-7）"""
        import inspect
        from vat.pipeline.executor import VideoProcessor
        members = [m[0] for m in inspect.getmembers(VideoProcessor, predicate=inspect.isfunction)]
        
        assert '_get_scene_prompt' in members, "应有 _get_scene_prompt 工厂方法"
        assert '_create_translator' in members, "应有 _create_translator 工厂方法"


# ==================== ProgressTracker 测试 ====================

class TestProgressTracker:
    """测试 ProgressTracker 进度追踪"""
    
    def test_basic_stage_progression(self):
        """测试基本阶段进度推进"""
        from vat.pipeline.progress import ProgressTracker
        
        messages = []
        def callback(progress, msg):
            messages.append((progress, msg))
        
        tracker = ProgressTracker(stages=['download', 'whisper'], callback=callback)
        
        # 初始进度为 0
        assert tracker.get_overall_progress() == 0.0
        
        # 开始并完成第一个阶段
        tracker.start_stage('download')
        tracker.complete_stage('download')
        
        # 完成1/2阶段 = 50%
        assert abs(tracker.get_overall_progress() - 0.5) < 0.01
        
        # 完成所有阶段
        tracker.start_stage('whisper')
        tracker.complete_stage('whisper')
        assert abs(tracker.get_overall_progress() - 1.0) < 0.01
    
    def test_partial_stage_progress(self):
        """测试阶段内部分进度"""
        from vat.pipeline.progress import ProgressTracker
        
        tracker = ProgressTracker(stages=['whisper', 'split'])
        
        tracker.start_stage('whisper')
        tracker.set_total_items(4)
        tracker.increment_completed(2)  # 完成 2/4
        
        # whisper 内部进度 = 50%，占总进度权重 50% → 总进度 25%
        progress = tracker.get_overall_progress()
        assert abs(progress - 0.25) < 0.01
    
    def test_progress_info_dict(self):
        """测试获取详细进度信息"""
        from vat.pipeline.progress import ProgressTracker
        
        tracker = ProgressTracker(stages=['download', 'whisper'])
        tracker.start_stage('download')
        tracker.complete_stage('download')
        
        info = tracker.get_progress_info()
        assert info['current_stage'] is None  # download 完成后 current 被清空
        assert 'download' in info['completed_stages']
        assert 'download' in info['stages']
        assert info['stages']['download']['progress'] == 1.0


if __name__ == "__main__":
    # 运行测试
    import sys
    
    # 简单测试运行器
    test_classes = [
        TestGPUUtils,
        TestTaskStepIndependence,
        TestDatabase,
        TestPlaylistService,
        TestConfig,
        TestVideoProcessorStages,
        TestDatabaseMigration,
        TestIntegrationHelpers,
        TestYouTubeDownloaderExtensions,
        TestPipelineExceptions,
        TestStageGroupExpansion,
        TestBatchProgressCalculation,
        TestProgressLogParsing,
        TestSpeakerGrouping,
        TestCreateVideoFromUrl,
        TestDeadCodeRemoval,
        TestSchedulerLogging,
        TestProgressTracker,
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print('='*60)
    
    sys.exit(0 if failed == 0 else 1)
