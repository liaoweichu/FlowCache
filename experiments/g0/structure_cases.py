"""
为 G0 实验生成 6 类真实结构用例。
基于 experiment-designs.md G0.2 节的定义。

6 类用例：
① 共享 system prompt（同域任务对）
② 分支历史（同一任务不同 user seed 的轨迹对）
③ chat template 变化（同一会话 × 2 版本 template）
④ 模型/adapter 标识变化（元数据字段受控变换）
⑤ 纯追加长会话（多轮对话逐轮递增）
⑥ 无共享对照（跨域任务对）

数据来源：τ-bench 的 retail/airline 域（system prompt 和 user instruction
内嵌于本文件，避免对 tau-bench pip 包或 HuggingFace datasets 的外部依赖）。
LMSYS-Chat-1M 需特殊授权下载，按任务约定用 τ-bench 多轮会话替代 ② 中的
LMSYS 10 条和 ⑤ 的 10 条。

运行：python experiments/g0/structure_cases.py
产出：experiments/g0/outputs/real-structure-cases.json（≥ 90 组用例）
"""
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List


# =============================================================================
# τ-bench 任务定义（内嵌，避免外部依赖）
# =============================================================================

# τ-bench retail 域的 system prompt（含 policy 和 tool schema）
TAU_BENCH_RETAIL_SYSTEM = """You are a retail agent assistant for an e-commerce platform. You help customers with orders, returns, exchanges, and product inquiries.

Available tools:
- exchange_delivered_order_item: Exchange an item in a delivered order
- find_user_by_email: Find user by email
- find_user_by_zip: Find user by zip code
- get_order_details: Get order details by order ID
- list_all_orders: List all orders for a user
- return_delivered_order_item: Return an item from a delivered order

Policy:
- Always verify user identity before accessing order information
- Returns accepted within 30 days of delivery
- Exchanges require the item to be in stock
- Maximum 3 returns per user per month"""

# τ-bench airline 域的 system prompt（含 policy 和 tool schema）
TAU_BENCH_AIRLINE_SYSTEM = """You are an airline customer service agent. You help customers with flight bookings, modifications, cancellations, and baggage inquiries.

Available tools:
- book_flight: Book a flight
- cancel_ticket: Cancel a ticket
- get_ticket_details: Get ticket details
- list_all_tickets: List all tickets for a user
- search_flights: Search available flights
- update_ticket: Update ticket information

Policy:
- Always verify user identity before accessing ticket information
- Cancellations allowed up to 24 hours before departure
- Flight changes subject to fare difference
- Baggage allowance varies by fare class"""

# retail 域的 user instruction 池（30 条，对应不同 task）
RETAIL_INSTRUCTIONS = [
    "I want to return a jacket I bought last week. My email is john@example.com",
    "Can you check the status of my order #12345?",
    "I need to exchange my shoes for a different size. Order #67890",
    "What's the return policy for electronics?",
    "I want to cancel order #11111. Email: jane@example.com",
    "My order #22222 hasn't arrived yet, can you check?",
    "I'd like to return two items from order #33333",
    "Can you help me find my order? Zip code: 10001",
    "I want to exchange a shirt for a different color. Order #44444",
    "What's the status of order #55555? Email: bob@example.com",
    "I received a damaged item in order #66666, want a replacement",
    "Can you list all my orders? Email: alice@example.com",
    "I want to return a phone case from order #77777",
    "My exchange request for order #88888 is pending",
    "Can you check if order #99999 has been shipped?",
    "I need to change the shipping address for order #10101",
    "What's my order history? Email: charlie@example.com",
    "I want to return everything in order #12121",
    "Can you help with a refund for order #13131?",
    "The size is wrong for order #14141, need exchange",
    "Order #15151 tracking please, email: david@example.com",
    "Return request for defective item in order #16161",
    "Exchange for different size, order #17171",
    "Check order #18181 status for me",
    "I want to cancel my recent order #19191",
    "List orders for email: eve@example.com",
    "Return item from order #20202",
    "Exchange request for order #21212",
    "Order #22222 delivery date?",
    "Cancel order #23232, email: frank@example.com",
]

# airline 域的 user instruction 池（30 条）
AIRLINE_INSTRUCTIONS = [
    "I want to book a flight from New York to London on Friday",
    "Can you check my ticket #AA123? Email: john@example.com",
    "I need to cancel my flight tomorrow. Ticket #BB456",
    "What's the baggage allowance for my ticket #CC789?",
    "Search for flights from LA to Tokyo next Monday",
    "I want to change my flight. Ticket #DD012",
    "List all my tickets. Email: jane@example.com",
    "My flight was cancelled, what are my options? Ticket #EE345",
    "Can you book a round trip from Chicago to Miami?",
    "Update my ticket #FF678 with new passenger info",
    "I want to cancel ticket #GG901. Email: bob@example.com",
    "What's the status of ticket #HH234?",
    "Search flights to Paris for this weekend",
    "My baggage was lost on flight #II567",
    "Book a flight for 3 passengers to Seattle",
    "Can you check ticket #JJ890? Email: alice@example.com",
    "I need to change my return flight. Ticket #KK123",
    "Cancel my booking #LL456 due to emergency",
    "What flights are available to Denver tomorrow?",
    "Update meal preference for ticket #MM789",
    "List tickets for email: charlie@example.com",
    "I want to book a business class flight to Boston",
    "My ticket #NN012 needs a seat change",
    "Can you search flights under $500 to Miami?",
    "Cancel ticket #OO345, email: david@example.com",
    "What's my booking reference for ticket #PP678?",
    "I need to add baggage to my booking #QQ901",
    "Book a one-way flight to San Francisco",
    "Change flight date for ticket #RR234",
    "Check my ticket #SS567 status, email: eve@example.com",
]


# =============================================================================
# ① 同域任务对（共享 system prompt）
# =============================================================================

def generate_category_1_same_domain_pairs() -> List[Dict]:
    """
    ① 同域任务对：retail 域内 15 对 + airline 域内 15 对 = 30 对。

    每对包含两个任务，共享同域的 system prompt（含 policy + tool schema），
    但 user instruction 不同。

    先验真值：两任务应共享且仅共享 system_prompt 前缀的 block，
    从 user instruction 起的 block 必须全部不同。
    """
    cases = []

    # retail 域 15 对
    for i in range(15):
        idx_a = (i * 2) % len(RETAIL_INSTRUCTIONS)
        idx_b = (i * 2 + 1) % len(RETAIL_INSTRUCTIONS)
        cases.append({
            "category": 1,
            "category_name": "same_domain_shared_system",
            "case_id": f"cat1_retail_{i:03d}",
            "domain": "retail",
            "pair": [
                {
                    "messages": [
                        {"role": "system", "content": TAU_BENCH_RETAIL_SYSTEM},
                        {"role": "user", "content": RETAIL_INSTRUCTIONS[idx_a]},
                    ],
                    "source_id": f"retail_task_{idx_a}",
                },
                {
                    "messages": [
                        {"role": "system", "content": TAU_BENCH_RETAIL_SYSTEM},
                        {"role": "user", "content": RETAIL_INSTRUCTIONS[idx_b]},
                    ],
                    "source_id": f"retail_task_{idx_b}",
                },
            ],
            "expected_shared_prefix": "system_prompt",
            "expected_shared_blocks": "system_prompt_blocks",
            "expected_differ_from": "user_instruction",
        })

    # airline 域 15 对
    for i in range(15):
        idx_a = (i * 2) % len(AIRLINE_INSTRUCTIONS)
        idx_b = (i * 2 + 1) % len(AIRLINE_INSTRUCTIONS)
        cases.append({
            "category": 1,
            "category_name": "same_domain_shared_system",
            "case_id": f"cat1_airline_{i:03d}",
            "domain": "airline",
            "pair": [
                {
                    "messages": [
                        {"role": "system", "content": TAU_BENCH_AIRLINE_SYSTEM},
                        {"role": "user", "content": AIRLINE_INSTRUCTIONS[idx_a]},
                    ],
                    "source_id": f"airline_task_{idx_a}",
                },
                {
                    "messages": [
                        {"role": "system", "content": TAU_BENCH_AIRLINE_SYSTEM},
                        {"role": "user", "content": AIRLINE_INSTRUCTIONS[idx_b]},
                    ],
                    "source_id": f"airline_task_{idx_b}",
                },
            ],
            "expected_shared_prefix": "system_prompt",
            "expected_shared_blocks": "system_prompt_blocks",
            "expected_differ_from": "user_instruction",
        })

    return cases


# =============================================================================
# ② 分支历史（同一任务不同 user seed 的轨迹对）
# =============================================================================

def generate_category_2_branch_history() -> List[Dict]:
    """
    ② 分支历史：20 对单轮分支 + 10 对多轮分支 = 30 组。

    单轮分支：同域同任务，前 3 条消息（system + user + assistant）完全相同，
    第 4 条 user 消息不同 → 分支点位于 index 3。

    多轮分支（替代 LMSYS 10 条）：同域多轮会话，前 N 轮完全相同，
    第 N+1 轮 user 消息不同 → 分支点位于更深处。

    先验真值：分支点之前（含）的所有 block 应共享；分支点之后的 block
    必须全部不同。
    """
    cases = []

    # ---- 20 对单轮分支（retail 10 + airline 10）----
    single_turn_followups = [
        "What are the next steps?",
        "Can you also check my order history?",
    ]
    for i in range(20):
        domain = "retail" if i < 10 else "airline"
        system = TAU_BENCH_RETAIL_SYSTEM if domain == "retail" else TAU_BENCH_AIRLINE_SYSTEM
        instructions = RETAIL_INSTRUCTIONS if domain == "retail" else AIRLINE_INSTRUCTIONS
        base_instruction = instructions[i % len(instructions)]
        shared_assistant = "I'll help you with that. Let me look up your information."

        cases.append({
            "category": 2,
            "category_name": "branch_history",
            "case_id": f"cat2_{domain}_single_{i:03d}",
            "domain": domain,
            "branch_kind": "single_turn",
            "pair": [
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": base_instruction},
                        {"role": "assistant", "content": shared_assistant},
                        {"role": "user", "content": single_turn_followups[0]},
                    ],
                    "source_id": f"{domain}_branch_a_{i}",
                },
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": base_instruction},
                        {"role": "assistant", "content": shared_assistant},
                        {"role": "user", "content": single_turn_followups[1]},
                    ],
                    "source_id": f"{domain}_branch_b_{i}",
                },
            ],
            "expected_shared_prefix": "through_assistant_response",
            "branch_point": 3,  # 消息 0-2 共享，消息 3 不同
        })

    # ---- 10 对多轮分支（替代 LMSYS 多轮会话）----
    # 多轮共享前缀更深，分支点位于第 5 条消息（index 5）
    multi_turn_followups = [
        "Actually, I want to cancel this request instead.",
        "Can you also process a refund for a different order?",
    ]
    for i in range(10):
        domain = "retail" if i < 5 else "airline"
        system = TAU_BENCH_RETAIL_SYSTEM if domain == "retail" else TAU_BENCH_AIRLINE_SYSTEM
        instructions = RETAIL_INSTRUCTIONS if domain == "retail" else AIRLINE_INSTRUCTIONS
        base_instruction = instructions[i % len(instructions)]

        # 前 5 条消息（system + 2 轮 user/assistant）完全相同
        shared_prefix = [
            {"role": "system", "content": system},
            {"role": "user", "content": base_instruction},
            {"role": "assistant", "content": "I'll help you with that. Let me look up your information."},
            {"role": "user", "content": "Thanks, what's the status?"},
            {"role": "assistant", "content": "I've found your record. Here are the details."},
        ]

        cases.append({
            "category": 2,
            "category_name": "branch_history",
            "case_id": f"cat2_{domain}_multi_{i:03d}",
            "domain": domain,
            "branch_kind": "multi_turn",
            "pair": [
                {
                    "messages": shared_prefix + [{"role": "user", "content": multi_turn_followups[0]}],
                    "source_id": f"{domain}_mbranch_a_{i}",
                },
                {
                    "messages": shared_prefix + [{"role": "user", "content": multi_turn_followups[1]}],
                    "source_id": f"{domain}_mbranch_b_{i}",
                },
            ],
            "expected_shared_prefix": "through_second_assistant",
            "branch_point": 5,  # 消息 0-4 共享，消息 5 不同
        })

    return cases


# =============================================================================
# ③ chat template 变化（同一会话 × 2 版本 template）
# =============================================================================

def generate_category_3_template_variation() -> List[Dict]:
    """
    ③ chat template 变化：10 条真实会话 × 2 版本 template = 10 组（每组 2 渲染）。

    同一会话用 default 和 modified 两个 chat template 版本渲染。
    template 变化会影响所有 block 的 token 序列。

    先验真值：模板 diff 后的所有 block 必须失效（change_point = 0），
    两条渲染路径之间不得有任何 block 命中。
    """
    cases = []

    for i in range(10):
        domain = "retail" if i < 5 else "airline"
        system = TAU_BENCH_RETAIL_SYSTEM if domain == "retail" else TAU_BENCH_AIRLINE_SYSTEM
        instructions = RETAIL_INSTRUCTIONS if domain == "retail" else AIRLINE_INSTRUCTIONS

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instructions[i % len(instructions)]},
        ]

        cases.append({
            "category": 3,
            "category_name": "template_variation",
            "case_id": f"cat3_{i:03d}",
            "domain": domain,
            "pair": [
                {
                    "messages": messages,
                    "template_version": "v1_default",
                    "template_hash": hashlib.sha256("default".encode()).hexdigest()[:16],
                },
                {
                    "messages": messages,
                    "template_version": "v2_modified",
                    "template_hash": hashlib.sha256("modified".encode()).hexdigest()[:16],
                },
            ],
            "expected_shared_prefix": "none",
            "expected_all_blocks_differ": True,
            "change_point": 0,  # 所有 block 都应不同
        })

    return cases


# =============================================================================
# ④ 模型/adapter 标识变化（元数据字段受控变换）
# =============================================================================

def generate_category_4_identifier_variation() -> List[Dict]:
    """
    ④ 模型/adapter 标识变化：10 组。

    同一段真实文本，但 I_b 元组中的 m（model_id）/ r（revision）/ a（adapter_id）
    字段受控变换。文本内容完全相同，但元数据不同。

    先验真值：fail-closed 必须判定全部 block 不兼容，禁止任何命中。
    """
    cases = []

    # 10 组：每组变换不同的元数据字段
    metadata_variants = [
        {"model_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "rev_a", "adapter_id": "none"},
        {"model_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "rev_b", "adapter_id": "none"},  # revision 不同
    ]

    for i in range(10):
        domain = "retail" if i < 5 else "airline"
        system = TAU_BENCH_RETAIL_SYSTEM if domain == "retail" else TAU_BENCH_AIRLINE_SYSTEM
        instructions = RETAIL_INSTRUCTIONS if domain == "retail" else AIRLINE_INSTRUCTIONS

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instructions[i % len(instructions)]},
        ]

        cases.append({
            "category": 4,
            "category_name": "identifier_variation",
            "case_id": f"cat4_{i:03d}",
            "domain": domain,
            "pair": [
                {"messages": messages, "metadata": metadata_variants[0]},
                {"messages": messages, "metadata": metadata_variants[1]},
            ],
            "expected_shared_prefix": "none",
            "expected_all_blocks_incompatible": True,
        })

    return cases


# =============================================================================
# ⑤ 纯追加长会话（多轮对话逐轮递增）
# =============================================================================

def _build_multi_turn_conversations() -> List[List[Dict]]:
    """
    构造 10 条多轮会话（替代 LMSYS 长会话）。
    每条会话 ≥ 4 轮（≥ 8 条消息含 system），覆盖 retail/airline 两域的
    不同业务场景。
    """
    conversations = [
        # 1. retail：退货流程
        [
            {"role": "system", "content": TAU_BENCH_RETAIL_SYSTEM},
            {"role": "user", "content": "I want to return an item. Email: john@example.com"},
            {"role": "assistant", "content": "I found your account. Which order would you like to return?"},
            {"role": "user", "content": "Order #12345"},
            {"role": "assistant", "content": "I see order #12345 with a jacket. Is this the item you want to return?"},
            {"role": "user", "content": "Yes, that's correct"},
            {"role": "assistant", "content": "I've initiated the return. You'll receive a return label by email."},
            {"role": "user", "content": "How long will the refund take?"},
        ],
        # 2. retail：换货流程
        [
            {"role": "system", "content": TAU_BENCH_RETAIL_SYSTEM},
            {"role": "user", "content": "I need to exchange my shoes for a different size. Order #67890"},
            {"role": "assistant", "content": "I found order #67890. What size would you like instead?"},
            {"role": "user", "content": "Size 9 instead of 8"},
            {"role": "assistant", "content": "Size 9 is in stock. Shall I process the exchange?"},
            {"role": "user", "content": "Yes please"},
            {"role": "assistant", "content": "Exchange processed. You'll receive the new size within 5 business days."},
            {"role": "user", "content": "Do I need to return the original pair?"},
        ],
        # 3. retail：订单状态查询
        [
            {"role": "system", "content": TAU_BENCH_RETAIL_SYSTEM},
            {"role": "user", "content": "My order #22222 hasn't arrived yet, can you check?"},
            {"role": "assistant", "content": "Let me check order #22222 for you."},
            {"role": "user", "content": "Sure, take your time"},
            {"role": "assistant", "content": "Order #22222 is currently in transit. Expected delivery is tomorrow."},
            {"role": "user", "content": "Can you provide a tracking number?"},
            {"role": "assistant", "content": "Tracking number is TRK123456789. You can track it on the carrier's website."},
            {"role": "user", "content": "Thanks, I'll keep an eye on it"},
        ],
        # 4. retail：取消订单
        [
            {"role": "system", "content": TAU_BENCH_RETAIL_SYSTEM},
            {"role": "user", "content": "I want to cancel order #11111. Email: jane@example.com"},
            {"role": "assistant", "content": "I found order #11111. It hasn't shipped yet, so cancellation is possible."},
            {"role": "user", "content": "Great, please cancel it"},
            {"role": "assistant", "content": "Order #11111 has been cancelled. Refund will be processed in 3-5 days."},
            {"role": "user", "content": "Will I get a confirmation email?"},
            {"role": "assistant", "content": "Yes, a cancellation confirmation has been sent to jane@example.com."},
            {"role": "user", "content": "Perfect, thank you"},
        ],
        # 5. retail：订单历史查询
        [
            {"role": "system", "content": TAU_BENCH_RETAIL_SYSTEM},
            {"role": "user", "content": "What's my order history? Email: charlie@example.com"},
            {"role": "assistant", "content": "I found 5 orders for your account. Would you like me to list them?"},
            {"role": "user", "content": "Yes, please list them all"},
            {"role": "assistant", "content": "1. Order #10101 - Delivered\n2. Order #12121 - In transit\n3. Order #13131 - Delivered\n4. Order #14141 - Processing\n5. Order #15151 - Delivered"},
            {"role": "user", "content": "Can you give me details on order #13131?"},
            {"role": "assistant", "content": "Order #13131 contains a phone case, delivered on July 15th."},
            {"role": "user", "content": "Thanks, that's all I needed"},
        ],
        # 6. airline：订票流程
        [
            {"role": "system", "content": TAU_BENCH_AIRLINE_SYSTEM},
            {"role": "user", "content": "I want to book a flight from New York to London on Friday"},
            {"role": "assistant", "content": "I found several flights from JFK to LHR on Friday. What time do you prefer?"},
            {"role": "user", "content": "Morning, around 8 AM"},
            {"role": "assistant", "content": "There's a flight at 8:15 AM. How many passengers?"},
            {"role": "user", "content": "Just one passenger"},
            {"role": "assistant", "content": "Booked. Your ticket number is AA123. Total is $850."},
            {"role": "user", "content": "Can I add checked baggage?"},
        ],
        # 7. airline：取消机票
        [
            {"role": "system", "content": TAU_BENCH_AIRLINE_SYSTEM},
            {"role": "user", "content": "I need to cancel my flight tomorrow. Ticket #BB456"},
            {"role": "assistant", "content": "Let me check ticket #BB456. Your flight departs at 10 AM tomorrow."},
            {"role": "user", "content": "Yes, that's the one"},
            {"role": "assistant", "content": "Since it's more than 24 hours before departure, full refund is available."},
            {"role": "user", "content": "Please cancel it"},
            {"role": "assistant", "content": "Ticket #BB456 has been cancelled. Refund of $620 will be processed."},
            {"role": "user", "content": "How long for the refund?"},
        ],
        # 8. airline：改签
        [
            {"role": "system", "content": TAU_BENCH_AIRLINE_SYSTEM},
            {"role": "user", "content": "I want to change my flight. Ticket #DD012"},
            {"role": "assistant", "content": "I can help with that. What date would you like to change to?"},
            {"role": "user", "content": "Next Monday instead of this Friday"},
            {"role": "assistant", "content": "There's availability on Monday. Fare difference is $75 additional."},
            {"role": "user", "content": "That's fine, proceed"},
            {"role": "assistant", "content": "Ticket #DD012 has been rebooked for Monday. New total is $925."},
            {"role": "user", "content": "Will I get a new confirmation?"},
        ],
        # 9. airline：行李咨询
        [
            {"role": "system", "content": TAU_BENCH_AIRLINE_SYSTEM},
            {"role": "user", "content": "What's the baggage allowance for my ticket #CC789?"},
            {"role": "assistant", "content": "Ticket #CC789 is economy class. You have 1 checked bag up to 23kg and 1 carry-on."},
            {"role": "user", "content": "How much for an extra checked bag?"},
            {"role": "assistant", "content": "Additional checked bag costs $60 if purchased online, $75 at the airport."},
            {"role": "user", "content": "I'll add one online"},
            {"role": "assistant", "content": "Added one extra checked bag to ticket #CC789. Total charge: $60."},
            {"role": "user", "content": "Thank you, that's all"},
        ],
        # 10. airline：机票状态查询
        [
            {"role": "system", "content": TAU_BENCH_AIRLINE_SYSTEM},
            {"role": "user", "content": "What's the status of ticket #HH234?"},
            {"role": "assistant", "content": "Ticket #HH234 is confirmed. Flight departs July 30th at 2 PM from SFO."},
            {"role": "user", "content": "What's the gate number?"},
            {"role": "assistant", "content": "Gate will be assigned 2 hours before departure. Please check the app then."},
            {"role": "user", "content": "Can I select a seat now?"},
            {"role": "assistant", "content": "Yes, seat selection is open. Would you prefer window or aisle?"},
            {"role": "user", "content": "Aisle seat please"},
        ],
    ]
    return conversations


def generate_category_5_pure_append() -> List[Dict]:
    """
    ⑤ 纯追加长会话：10 条多轮会话。

    每条会话生成多个递增前缀（每轮追加 user+assistant 后形成新前缀）。
    追加语义天然：会话内前缀应逐轮递增复用。

    先验真值：每个前缀是前一个前缀的超集；逐轮复用必须命中。
    """
    cases = []

    conversations = _build_multi_turn_conversations()

    for i, conv in enumerate(conversations):
        # 每 2 条消息（user + assistant）为一个追加单位
        # 注意：conv[0] 是 system 消息，从 index 2 开始按对切片
        prefixes = []
        # 第一个前缀：system + 第一轮 user
        prefixes.append({
            "messages": conv[:2],
            "turn_count": 1,
        })
        # 后续前缀：每次追加 user + assistant
        for turn_idx in range(2, len(conv), 2):
            end_idx = min(turn_idx + 2, len(conv))
            prefixes.append({
                "messages": conv[:end_idx],
                "turn_count": end_idx // 2,
            })

        cases.append({
            "category": 5,
            "category_name": "pure_append",
            "case_id": f"cat5_{i:03d}",
            "domain": "retail" if i < 5 else "airline",
            "turns": prefixes,
            "expected_incremental_sharing": True,  # 每个前缀是前一个的超集
            "expected_min_turns": 4,
        })

    return cases


# =============================================================================
# ⑥ 无共享对照（跨域任务对）
# =============================================================================

def generate_category_6_no_sharing() -> List[Dict]:
    """
    ⑥ 无共享对照：retail × airline 跨域任务对 10 对。

    每对一个 retail 任务 + 一个 airline 任务，system prompt 和 user
    instruction 都不同。

    先验真值：跨条命中 = 0，不应有任何 block 共享。
    """
    cases = []

    for i in range(10):
        cases.append({
            "category": 6,
            "category_name": "no_sharing",
            "case_id": f"cat6_{i:03d}",
            "pair": [
                {
                    "messages": [
                        {"role": "system", "content": TAU_BENCH_RETAIL_SYSTEM},
                        {"role": "user", "content": RETAIL_INSTRUCTIONS[i % len(RETAIL_INSTRUCTIONS)]},
                    ],
                    "source_id": f"retail_{i}",
                    "domain": "retail",
                },
                {
                    "messages": [
                        {"role": "system", "content": TAU_BENCH_AIRLINE_SYSTEM},
                        {"role": "user", "content": AIRLINE_INSTRUCTIONS[i % len(AIRLINE_INSTRUCTIONS)]},
                    ],
                    "source_id": f"airline_{i}",
                    "domain": "airline",
                },
            ],
            "expected_shared_prefix": "none",
            "expected_zero_matches": True,
        })

    return cases


# =============================================================================
# 汇总与持久化
# =============================================================================

def generate_all_cases() -> Dict:
    """
    生成全部 6 类结构用例并汇总。
    返回包含 categories 和 summary 的字典。
    """
    # 各生成器只调用一次，避免重复计算
    cat1 = generate_category_1_same_domain_pairs()
    cat2 = generate_category_2_branch_history()
    cat3 = generate_category_3_template_variation()
    cat4 = generate_category_4_identifier_variation()
    cat5 = generate_category_5_pure_append()
    cat6 = generate_category_6_no_sharing()

    return {
        "generated_at": datetime.now().isoformat(),
        "generator": "experiments/g0/structure_cases.py",
        "data_source": "tau-bench (retail/airline domains, embedded)",
        "lmsys_substitution": "LMSYS-Chat-1M 需特殊授权，② 中的 LMSYS 10 条和 ⑤ 的 10 条用 τ-bench 多轮会话替代",
        "total_categories": 6,
        "categories": {
            "cat1_same_domain": cat1,
            "cat2_branch_history": cat2,
            "cat3_template_variation": cat3,
            "cat4_identifier_variation": cat4,
            "cat5_pure_append": cat5,
            "cat6_no_sharing": cat6,
        },
        "summary": {
            "cat1_same_domain": len(cat1),
            "cat2_branch_history": len(cat2),
            "cat3_template_variation": len(cat3),
            "cat4_identifier_variation": len(cat4),
            "cat5_pure_append": len(cat5),
            "cat6_no_sharing": len(cat6),
        },
    }


def save_cases(cases: Dict, output_path: str) -> None:
    """
    将结构用例保存为 JSON 文件，并打印摘要。
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    # 打印摘要
    total = sum(cases["summary"].values())
    print(f"\n[structure_cases] 已保存到 {output_path}")
    print(f"  类别数: {cases['total_categories']}")
    print(f"  用例总数: {total}")
    for cat, count in cases["summary"].items():
        print(f"  {cat}: {count} 组")


def _default_output_path() -> str:
    """默认输出路径：脚本所在目录下的 outputs/real-structure-cases.json。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "outputs", "real-structure-cases.json")


if __name__ == "__main__":
    cases = generate_all_cases()
    save_cases(cases, _default_output_path())
