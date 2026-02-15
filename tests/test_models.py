"""
models.py 单元测试

测试阶段定义、依赖关系、阶段组展开、依赖解析等核心逻辑。
"""
import pytest
from vat.models import (
    TaskStep, TaskStatus, SourceType,
    STAGE_GROUPS, STAGE_DEPENDENCIES, DEFAULT_STAGE_SEQUENCE,
    expand_stage_group, get_required_stages,
    Video, Task, Playlist,
)


class TestDefaultStageSequence:
    """DEFAULT_STAGE_SEQUENCE 完整性"""

    def test_length_is_7(self):
        assert len(DEFAULT_STAGE_SEQUENCE) == 7

    def test_first_is_download_last_is_upload(self):
        assert DEFAULT_STAGE_SEQUENCE[0] == TaskStep.DOWNLOAD
        assert DEFAULT_STAGE_SEQUENCE[-1] == TaskStep.UPLOAD

    def test_all_taskstep_members_covered(self):
        """每个 TaskStep 枚举成员都出现在默认序列中"""
        assert set(DEFAULT_STAGE_SEQUENCE) == set(TaskStep)

    def test_no_duplicates(self):
        assert len(DEFAULT_STAGE_SEQUENCE) == len(set(DEFAULT_STAGE_SEQUENCE))


class TestStageDependencies:
    """阶段依赖关系完整性"""

    def test_every_step_has_dependency_entry(self):
        """每个 TaskStep 都在 STAGE_DEPENDENCIES 中有定义"""
        for step in TaskStep:
            assert step in STAGE_DEPENDENCIES, f"{step} 缺少依赖定义"

    def test_download_has_no_deps(self):
        assert STAGE_DEPENDENCIES[TaskStep.DOWNLOAD] == []

    def test_non_download_steps_have_deps(self):
        """除 DOWNLOAD 外，每个阶段都至少有一个前置依赖"""
        for step in DEFAULT_STAGE_SEQUENCE[1:]:
            assert len(STAGE_DEPENDENCIES[step]) > 0, f"{step} 应有前置依赖"

    def test_dependency_chain_is_linear(self):
        """依赖链严格线性：每个阶段仅依赖前一个阶段"""
        for i in range(1, len(DEFAULT_STAGE_SEQUENCE)):
            current = DEFAULT_STAGE_SEQUENCE[i]
            deps = STAGE_DEPENDENCIES[current]
            assert deps == [DEFAULT_STAGE_SEQUENCE[i - 1]], \
                f"{current} 的依赖应为 [{DEFAULT_STAGE_SEQUENCE[i - 1]}]，实际 {deps}"

    def test_no_circular_dependency(self):
        """不存在循环依赖"""
        visited = set()
        for step in DEFAULT_STAGE_SEQUENCE:
            for dep in STAGE_DEPENDENCIES[step]:
                assert dep in visited, f"{step} 依赖了尚未出现的 {dep}"
            visited.add(step)


class TestStageGroups:
    """阶段组定义"""

    def test_asr_group(self):
        assert STAGE_GROUPS["asr"] == [TaskStep.WHISPER, TaskStep.SPLIT]

    def test_translate_group(self):
        assert STAGE_GROUPS["translate"] == [TaskStep.OPTIMIZE, TaskStep.TRANSLATE]

    def test_single_step_groups(self):
        assert STAGE_GROUPS["download"] == [TaskStep.DOWNLOAD]
        assert STAGE_GROUPS["embed"] == [TaskStep.EMBED]
        assert STAGE_GROUPS["upload"] == [TaskStep.UPLOAD]

    def test_all_steps_covered_by_groups(self):
        """所有阶段组合起来应覆盖全部 TaskStep"""
        all_steps = []
        for steps in STAGE_GROUPS.values():
            all_steps.extend(steps)
        assert set(all_steps) == set(TaskStep)


class TestExpandStageGroup:
    """expand_stage_group 展开逻辑"""

    def test_expand_group_name(self):
        assert expand_stage_group("asr") == [TaskStep.WHISPER, TaskStep.SPLIT]

    def test_single_step_priority_over_group(self):
        """当名称同时匹配单步和组时，优先匹配单步（解决 'translate' 歧义）"""
        # 'translate' 既是 TaskStep.TRANSLATE 的值，也是 STAGE_GROUPS 的 key
        # 应优先匹配单步，避免 -s translate 意外带上 optimize
        assert expand_stage_group("translate") == [TaskStep.TRANSLATE]

    def test_expand_single_step(self):
        assert expand_stage_group("whisper") == [TaskStep.WHISPER]
        assert expand_stage_group("download") == [TaskStep.DOWNLOAD]

    def test_case_insensitive(self):
        assert expand_stage_group("ASR") == [TaskStep.WHISPER, TaskStep.SPLIT]
        assert expand_stage_group("Download") == [TaskStep.DOWNLOAD]

    def test_unknown_raises_valueerror(self):
        with pytest.raises(ValueError, match="未知"):
            expand_stage_group("nonexistent")

    def test_expand_pure_group_names(self):
        """无同名单步的组名能正常展开为整组"""
        # 仅测试不与 TaskStep value 冲突的组名
        step_values = {s.value for s in TaskStep}
        for group_name, expected in STAGE_GROUPS.items():
            if group_name not in step_values:
                result = expand_stage_group(group_name)
                assert result == expected, f"组 '{group_name}' 展开不符合预期"

    def test_expand_all_step_values(self):
        """所有 TaskStep 的 value 都能作为单阶段展开"""
        for step in TaskStep:
            result = expand_stage_group(step.value)
            assert step in result


class TestGetRequiredStages:
    """get_required_stages 依赖解析"""

    def test_download_only(self):
        required = get_required_stages([TaskStep.DOWNLOAD])
        assert required == [TaskStep.DOWNLOAD]

    def test_whisper_includes_download(self):
        required = get_required_stages([TaskStep.WHISPER])
        assert required == [TaskStep.DOWNLOAD, TaskStep.WHISPER]

    def test_translate_includes_all_predecessors(self):
        required = get_required_stages([TaskStep.TRANSLATE])
        expected = [
            TaskStep.DOWNLOAD, TaskStep.WHISPER, TaskStep.SPLIT,
            TaskStep.OPTIMIZE, TaskStep.TRANSLATE,
        ]
        assert required == expected

    def test_upload_includes_everything(self):
        required = get_required_stages([TaskStep.UPLOAD])
        assert required == DEFAULT_STAGE_SEQUENCE

    def test_multiple_targets_deduplicated(self):
        """多个目标阶段的依赖应去重"""
        required = get_required_stages([TaskStep.SPLIT, TaskStep.TRANSLATE])
        # SPLIT 的依赖是 DOWNLOAD, WHISPER, SPLIT
        # TRANSLATE 的依赖是 DOWNLOAD, WHISPER, SPLIT, OPTIMIZE, TRANSLATE
        expected = [
            TaskStep.DOWNLOAD, TaskStep.WHISPER, TaskStep.SPLIT,
            TaskStep.OPTIMIZE, TaskStep.TRANSLATE,
        ]
        assert required == expected

    def test_order_matches_default_sequence(self):
        """返回结果顺序与 DEFAULT_STAGE_SEQUENCE 一致"""
        required = get_required_stages([TaskStep.EMBED])
        for i in range(len(required) - 1):
            idx_a = DEFAULT_STAGE_SEQUENCE.index(required[i])
            idx_b = DEFAULT_STAGE_SEQUENCE.index(required[i + 1])
            assert idx_a < idx_b

    def test_empty_input(self):
        assert get_required_stages([]) == []


class TestDataModels:
    """数据模型基本行为"""

    def test_video_source_type_coercion(self):
        """Video 的 source_type 字符串应自动转为枚举"""
        v = Video(id="test", source_type="youtube", source_url="http://example.com")
        assert v.source_type == SourceType.YOUTUBE

    def test_task_step_coercion(self):
        """Task 的 step 字符串应自动转为枚举"""
        t = Task(video_id="v1", step="whisper", status="pending")
        assert t.step == TaskStep.WHISPER
        assert t.status == TaskStatus.PENDING

    def test_video_auto_timestamps(self):
        """Video 创建时自动设置时间戳"""
        v = Video(id="test", source_type=SourceType.YOUTUBE, source_url="http://example.com")
        assert v.created_at is not None
        assert v.updated_at is not None

    def test_playlist_defaults(self):
        p = Playlist(id="PL1", title="Test", source_url="http://example.com")
        assert p.video_count == 0
        assert p.metadata == {}
        assert p.created_at is not None
