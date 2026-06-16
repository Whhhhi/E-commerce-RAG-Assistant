import pytest
from app.core.intent_router import IntentRouter, Intent


class TestIntentRouter:
    def test_keyword_product_inquiry(self):
        router = IntentRouter(use_llm_fallback=False)
        assert router.classify("这款手机多少钱？") == Intent.PRODUCT_INQUIRY
        assert router.classify("有货吗？") == Intent.PRODUCT_INQUIRY
        assert router.classify("推荐一款手机") == Intent.PRODUCT_INQUIRY

    def test_keyword_return_exchange(self):
        router = IntentRouter(use_llm_fallback=False)
        assert router.classify("怎么退货？") == Intent.RETURN_EXCHANGE
        assert router.classify("7天无理由退换货") == Intent.RETURN_EXCHANGE
        assert router.classify("质量问题要退款") == Intent.RETURN_EXCHANGE

    def test_keyword_order_tracking(self):
        router = IntentRouter(use_llm_fallback=False)
        assert router.classify("我的订单到哪了？") == Intent.ORDER_TRACKING
        assert router.classify("快递单号是多少") == Intent.ORDER_TRACKING
        assert router.classify("什么时候发货") == Intent.ORDER_TRACKING

    def test_chitchat_fallback(self):
        router = IntentRouter(use_llm_fallback=False)
        assert router.classify("你好") == Intent.CHITCHAT
        assert router.classify("今天天气不错") == Intent.CHITCHAT

    def test_route_info(self):
        router = IntentRouter(use_llm_fallback=False)
        
        route = router.route("这款手机多少钱？")
        assert route["intent"] == Intent.PRODUCT_INQUIRY
        assert route["use_rag"] is True
        assert route["knowledge_base"] == "product"
        
        route = router.route("怎么退货？")
        assert route["intent"] == Intent.RETURN_EXCHANGE
        assert route["use_rag"] is True
        assert route["knowledge_base"] == "policy"
        
        route = router.route("我的订单到哪了？")
        assert route["intent"] == Intent.ORDER_TRACKING
        assert route["use_tool"] is True
        
        route = router.route("你好")
        assert route["intent"] == Intent.CHITCHAT
        assert route["use_rag"] is False
        assert route["use_tool"] is False