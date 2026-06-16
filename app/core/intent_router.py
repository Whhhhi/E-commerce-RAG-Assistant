import re
from enum import Enum
from typing import Optional

from app.services.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate


class Intent(str, Enum):
    PRODUCT_INQUIRY = "product_inquiry"
    RETURN_EXCHANGE = "return_exchange"
    ORDER_TRACKING = "order_tracking"
    CHITCHAT = "chitchat"


KEYWORD_RULES: dict[Intent, list[re.Pattern]] = {
    Intent.PRODUCT_INQUIRY: [
        re.compile(r"(价格|多少钱|售价|优惠|折扣|包邮|规格|参数|尺寸|颜色|版本|保修期)"),
        re.compile(r"(有货|库存|现货|预售|什么时候.*上)"),
        re.compile(r"(推荐|哪款|适合|对比|区别|好不好|好用吗)"),
    ],
    Intent.RETURN_EXCHANGE: [
        re.compile(r"(退货|退款|换货|退换|售后|维修|返修)"),
        re.compile(r"(无理由|拒收|取消订单|质量问题|补偿)"),
        re.compile(r"(退钱|退差价|补发|少发|错发)"),
    ],
    Intent.ORDER_TRACKING: [
        re.compile(r"(订单|物流|快递|发货|配送|签收|运单号)"),
        re.compile(r"(到哪|多久.*到|什么时候.*到|配送中)"),
    ],
}


INTENT_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个电商客服意图分类器。请判断用户问题的意图类别，只返回一个类别名称。\n\n"
        "类别定义：\n"
        "- product_inquiry: 商品咨询（问价格、规格、推荐、库存等）\n"
        "- return_exchange: 退换货/售后（退货、退款、换货、维修等）\n"
        "- order_tracking: 订单查询（物流、配送、发货时间等）\n"
        "- chitchat: 闲聊/问候（打招呼、感谢、非业务相关）\n\n"
        "只输出类别名，不要附加任何其他文字。",
    ),
    ("human", "{message}"),
])


UNKNOWN_INTENT_MSG = (
    "抱歉，我没能理解您的问题类型。您可以尝试以下方式：\n"
    "1. 咨询商品信息，例如「这款手机多少钱？」\n"
    "2. 查询退换货政策，例如「怎么退货？」\n"
    "3. 查询订单状态，例如「我的订单到哪了？」"
)


class IntentRouter:
    def __init__(self, use_llm_fallback: bool = True):
        self.use_llm_fallback = use_llm_fallback

    def classify(self, message: str) -> Intent:
        for intent, patterns in KEYWORD_RULES.items():
            for pattern in patterns:
                if pattern.search(message):
                    return intent

        if self.use_llm_fallback:
            try:
                chain = INTENT_CLASSIFY_PROMPT | get_llm(temperature=0)
                result = chain.invoke({"message": message}).content.strip().lower()

                for intent in Intent:
                    if intent.value in result:
                        return intent
            except Exception:
                pass

        return Intent.CHITCHAT

    def route(self, message: str) -> dict:
        intent = self.classify(message)

        routing_info = {
            "intent": intent,
            "use_rag": intent in (Intent.PRODUCT_INQUIRY, Intent.RETURN_EXCHANGE),
            "use_tool": intent == Intent.ORDER_TRACKING,
            "knowledge_base": (
                "product" if intent == Intent.PRODUCT_INQUIRY
                else "policy" if intent == Intent.RETURN_EXCHANGE
                else None
            ),
        }
        return routing_info