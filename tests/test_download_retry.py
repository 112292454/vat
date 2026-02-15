"""
测试下载器的网络错误分类逻辑

验证 _is_retryable_network_error 能正确区分：
- 可重试的瞬态网络错误（VPN/proxy 故障）
- 不可重试的 YouTube 限制（风控/cookie/视频不可用）
"""
import pytest
from vat.downloaders.youtube import _is_retryable_network_error


class TestRetryableErrorClassification:
    """测试错误分类：可重试 vs 不可重试"""
    
    # ==================== 可重试的网络错误 ====================
    
    def test_connection_reset(self):
        msg = "('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))"
        assert _is_retryable_network_error(msg) is True
    
    def test_connection_refused_proxy(self):
        msg = ("('Unable to connect to proxy', NewConnectionError("
               "'<urllib3.connection.HTTPSConnection object>: "
               "Failed to establish a new connection: [Errno 111] Connection refused'))")
        assert _is_retryable_network_error(msg) is True
    
    def test_proxy_error(self):
        msg = "ProxyError('Cannot connect to proxy')"
        assert _is_retryable_network_error(msg) is True
    
    def test_503_service_unavailable(self):
        msg = "HTTP Error 503: Service Temporarily Unavailable"
        assert _is_retryable_network_error(msg) is True
    
    def test_502_bad_gateway(self):
        msg = "HTTP Error 502: Bad Gateway"
        assert _is_retryable_network_error(msg) is True
    
    def test_network_unreachable(self):
        msg = "[Errno 101] Network is unreachable"
        assert _is_retryable_network_error(msg) is True
    
    def test_ssl_error(self):
        msg = "SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING]"
        assert _is_retryable_network_error(msg) is True
    
    def test_read_timeout(self):
        msg = "Read timed out. (read timeout=30)"
        assert _is_retryable_network_error(msg) is True
    
    def test_new_connection_error(self):
        msg = "NewConnectionError: Failed to establish a new connection"
        assert _is_retryable_network_error(msg) is True
    
    # ==================== 不可重试的 YouTube 限制 ====================
    
    def test_bot_detection(self):
        msg = ("Sign in to confirm you're not a bot. "
               "Use --cookies-from-browser or --cookies for the authentication.")
        assert _is_retryable_network_error(msg) is False
    
    def test_rate_limited(self):
        msg = ("Video unavailable. This content isn't available, try again later. "
               "Your account has been rate-limited by YouTube for up to an hour.")
        assert _is_retryable_network_error(msg) is False
    
    def test_video_private(self):
        msg = "This video is private"
        assert _is_retryable_network_error(msg) is False
    
    def test_video_unavailable(self):
        msg = "Video unavailable"
        assert _is_retryable_network_error(msg) is False
    
    def test_video_removed(self):
        msg = "This video has been removed by the uploader"
        assert _is_retryable_network_error(msg) is False
    
    def test_copyright(self):
        msg = "This video has been removed for copyright reasons"
        assert _is_retryable_network_error(msg) is False
    
    def test_account_terminated(self):
        msg = "This account has been terminated"
        assert _is_retryable_network_error(msg) is False
    
    # ==================== 未知错误（不重试） ====================
    
    def test_unknown_error_not_retried(self):
        msg = "Some completely unknown error happened"
        assert _is_retryable_network_error(msg) is False
    
    def test_empty_error(self):
        assert _is_retryable_network_error("") is False
    
    # ==================== 优先级：不可重试 > 可重试 ====================
    
    def test_rate_limit_with_connection_keyword(self):
        """即使错误消息中包含网络关键词，YouTube 限制模式应优先"""
        msg = ("Video unavailable. This content isn't available, try again later. "
               "Connection was reset but rate-limited")
        assert _is_retryable_network_error(msg) is False
