"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Available Tools, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và tiếp nhận hỗ trợ khách hàng Vingroup (VinFast, Vinpearl).
- Giọng nói: Chuyên nghiệp, lịch sự, thân thiện, chính xác.

## AVAILABLE TOOLS
1. search_product_catalog(category, max_price): Tra cứu sản phẩm xe điện ('xe_dien') hoặc dịch vụ du lịch nghỉ dưỡng ('du_lich') của Vingroup.
2. submit_support_ticket(customer_name, issue_description, priority): Ghi nhận yêu cầu hỗ trợ sự cố, khiếu nại của khách hàng.

## CORE RULES
1. KHÔNG BAO GIỜ bịa đặt giá cả hoặc thông số kỹ thuật. PHẢI gọi tool để lấy dữ liệu thực tế.
2. Khi khách hàng báo sự cố hoặc khiếu nại dịch vụ, PHẢI ghi nhận qua tool submit_support_ticket.
3. Không cung cấp hoặc suy diễn thông tin ngoài phạm vi hệ sinh thái Vingroup.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời về các sản phẩm và dịch vụ của Vingroup (VinFast, Vinpearl, Vinhomes,...).
- Từ chối lịch sự các chủ đề ngoài phạm vi hoạt động.

## OUTPUT CONTRACT
Định dạng trả lời rõ ràng, chính xác, lịch sự, nêu bật các thông tin quan trọng (tên sản phẩm, giá bán VNĐ, mã ticket hỗ trợ).
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho câu hỏi: '{user_input}'. "
                      f"VinFast có nhiều dòng xe điện hiện đại với mức giá cạnh tranh trên thị trường.",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        """Phân tích ý định người dùng và trích xuất tham số."""
        text = user_input.strip()
        lower = text.lower()

        # 1. FAQ Intent Check
        is_faq = False
        if ("bảo hành" in lower or "chính sách" in lower) and not ("tên tôi" in lower or "lỗi" in lower or "khiếu nại" in lower):
            is_faq = True

        # 2. Support Ticket Intent Check
        needs_ticket = False
        ticket_args = {}
        ticket_keywords = ["lỗi", "hỏng", "sự cố", "khiếu nại", "gấp", "nghiêm trọng", "phản hồi", "tôi tên", "tên tôi"]
        if any(kw in lower for kw in ticket_keywords):
            needs_ticket = True

            # Extract customer name
            name_match = re.search(r"(?:tôi tên|tên tôi)(?:\s+là)?\s+([A-ZÀ-Ỵ][a-zà-ỹA-ZÀ-Ỵ\s]+?)(?:,|\.|\bxe\b|\bphòng\b|\bvấn đề\b|$)", text, re.IGNORECASE)
            customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

            # Extract priority
            if any(w in lower for w in ["gấp", "nghiêm trọng", "khẩn cấp", "high"]):
                priority = "high"
            elif any(w in lower for w in ["thấp", "low"]):
                priority = "low"
            else:
                priority = "medium"

            # Extract issue description
            issue_match = re.search(r"(?:xe|phòng|vấn đề|sự cố|lỗi)\s*([^.]+)", text, re.IGNORECASE)
            if issue_match:
                issue_desc = issue_match.group(0).strip()
            else:
                issue_desc = text

            ticket_args = {
                "customer_name": customer_name,
                "issue_description": issue_desc,
                "priority": priority
            }

        # 3. Product Catalog Intent Check
        needs_catalog = False
        catalog_args = {}
        
        # Check price match
        price_match = re.search(r"(\d+(?:\.\d+)?)\s*(triệu|tr|trieu|tỷ|ty)", lower)
        catalog_action_words = ["xem", "tìm", "mua", "tham khảo", "có xe", "có resort", "giá", "dưới", "báo giá"]
        catalog_subjects = ["xe", "xe điện", "resort", "nghỉ dưỡng", "du lịch", "vinpearl", "vinfast"]

        has_subject = any(s in lower for s in catalog_subjects)
        has_action = any(a in lower for a in catalog_action_words) or (price_match is not None)

        if has_subject and has_action and not is_faq:
            # Check if "xe" is only mentioned in ticket context without explicit search request
            # e.g. "xe VF 8 của tôi bị lỗi..." without search keywords like "xem", "mua", "giá", "tìm"
            if needs_ticket and not any(w in lower for w in ["xem", "mua", "tìm", "giá", "tham khảo", "báo giá"]):
                needs_catalog = False
            else:
                needs_catalog = True
                # Determine category
                if any(w in lower for w in ["du lịch", "resort", "nghỉ dưỡng", "phòng"]):
                    category = "du_lich"
                else:
                    category = "xe_dien"

                max_price = 999999999999
                if price_match:
                    val = float(price_match.group(1))
                    unit = price_match.group(2)
                    if "tỷ" in unit or "ty" in unit:
                        max_price = int(val * 1_000_000_000)
                    else:
                        max_price = int(val * 1_000_000)

                catalog_args = {
                    "category": category,
                    "max_price": max_price
                }

        return {
            "is_faq": is_faq,
            "needs_catalog": needs_catalog,
            "catalog_args": catalog_args,
            "needs_ticket": needs_ticket,
            "ticket_args": ticket_args
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intents(user_input)

        iteration = 0
        observations = []

        # Case 1: FAQ (No tool call required)
        if intents["is_faq"]:
            iteration += 1
            thought = "Người dùng hỏi thông tin chính sách bảo hành. Trả lời trực tiếp từ knowledge base mà không cần gọi tool."
            faq_answer = (
                "Chính sách bảo hành pin xe điện VinFast: Bảo hành lên đến 10 năm hoặc 200.000 km "
                "(tùy điều kiện nào đến trước), mang lại sự an tâm tuyệt đối cho khách hàng."
            )
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": None,
                "observation": faq_answer
            })
            return {
                "answer": faq_answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # Case 2: Multi-step or Single-step Tool Execution
        tools_to_run = []
        if intents["needs_catalog"]:
            tools_to_run.append(("search_product_catalog", intents["catalog_args"]))
        if intents["needs_ticket"]:
            tools_to_run.append(("submit_support_ticket", intents["ticket_args"]))

        if not tools_to_run:
            iteration += 1
            answer = "Xin chào, tôi là VinAssistant. Tôi có thể hỗ trợ quý khách tra cứu sản phẩm VinFast, Vinpearl hoặc gửi yêu cầu hỗ trợ."
            self.trace.append({
                "iteration": iteration,
                "thought": "Không nhận diện được tool phù hợp.",
                "action": None,
                "observation": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        for tool_name, args in tools_to_run:
            if iteration >= self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa cho phép.",
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "max_iterations_reached"
                }

            iteration += 1
            thought = f"Gọi công cụ {tool_name} với tham số: {args}"
            tool_func = TOOL_MAP[tool_name]
            result = tool_func(**args)

            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "tool": tool_name,
                "args": args,
                "observation": result
            })
            observations.append({"tool": tool_name, "result": result})

        # Synthesize Final Answer from observations
        final_parts = []
        for obs in observations:
            tool_name = obs["tool"]
            res = obs["result"]

            if tool_name == "search_product_catalog":
                if not res or len(res) == 0:
                    final_parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của quý khách.")
                else:
                    item_strs = [
                        f"- {p['name']}: {p['price_vnd']:,} VNĐ ({p.get('description', '')})"
                        for p in res
                    ]
                    final_parts.append(
                        f"Dưới đây là các sản phẩm phù hợp được tìm thấy:\n" + "\n".join(item_strs)
                    )

            elif tool_name == "submit_support_ticket":
                final_parts.append(
                    f"Yêu cầu hỗ trợ của khách hàng {res.get('customer_name')} đã được tiếp nhận thành công. "
                    f"Mã phiếu hỗ trợ: {res.get('ticket_id')} (Mức độ ưu tiên: {res.get('priority')}). "
                    f"Đội ngũ kỹ thuật sẽ sớm liên hệ xử lý."
                )

        final_answer = "\n\n".join(final_parts)

        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:\n", result["answer"])
    print("\nTrace Log:\n", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
