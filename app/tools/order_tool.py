from typing import Optional


def query_order(message: str) -> str:
    """
    模拟订单查询工具
    实际应用中应调用外部订单查询 API
    """
    mock_orders = {
        "123456": {
            "order_id": "123456",
            "status": "已发货",
            "tracking_number": "SF1234567890",
            "estimated_delivery": "2026-06-18",
            "carrier": "顺丰速运",
        },
        "987654": {
            "order_id": "987654",
            "status": "配送中",
            "tracking_number": "JD9876543210",
            "estimated_delivery": "2026-06-16",
            "carrier": "京东物流",
        },
    }

    order_id = extract_order_id(message)
    if order_id and order_id in mock_orders:
        order = mock_orders[order_id]
        return (
            f"您的订单 {order['order_id']} 当前状态：{order['status']}\n"
            f"快递公司：{order['carrier']}\n"
            f"运单号：{order['tracking_number']}\n"
            f"预计送达：{order['estimated_delivery']}"
        )
    else:
        return "抱歉，我需要您提供订单号才能查询订单状态。请告诉我您的订单号。"


def extract_order_id(message: str) -> Optional[str]:
    """
    从消息中提取订单号（简单实现）
    """
    import re
    match = re.search(r"(\d{6,12})", message)
    if match:
        return match.group(1)
    return None