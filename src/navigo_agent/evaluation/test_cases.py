"""Curated evaluation test cases for the Navigo-Agent supervisor system.

Each case defines: query, expected_intent and expected_agents (the agents
that SHOULD run; None → not asserted). A case may also carry a `follow_up`
query, which is sent as a second turn of the same conversation — the
second answer is the one judged for quality.
"""

# ── Test Cases ───────────────────────────────────────────────────────

EVALUATION_CASES = [
    {
        "id": "full_trip_001",
        "category": "full_trip",
        "query": "Plan a complete 7 days Japan trip from Bangladesh including flights, hotels and sightseeing under 2 lakhs.",
        "expected_intent": "full_trip",
        "expected_agents": ["flight_agent", "hotel_agent", "weather_agent", "itinerary_agent"],
    },
    {
        "id": "weather_only_001",
        "category": "weather",
        "query": "What is the weather like in Tokyo right now?",
        "expected_intent": "weather",
        "expected_agents": ["weather_agent"],
    },
    {
        "id": "weather_only_002",
        "category": "weather",
        "query": "Tell me the forecast for Paris next week",
        "expected_intent": "weather",
        "expected_agents": ["weather_agent"],
    },
    {
        "id": "flight_only_001",
        "category": "flights",
        "query": "Find me flights from Dhaka to Dubai",
        "expected_intent": "flight",
        "expected_agents": ["flight_agent"],
    },
    {
        "id": "hotel_only_001",
        "category": "hotel",
        "query": "Find budget hotels in Bangkok near Sukhumvit",
        "expected_intent": "hotel",
        "expected_agents": ["hotel_agent"],
    },
    {
        "id": "full_trip_002",
        "category": "full_trip",
        "query": "Plan a 5-day trip to Singapore with family, mid-range budget",
        "expected_intent": "full_trip",
        "expected_agents": ["flight_agent", "hotel_agent", "weather_agent", "itinerary_agent"],
    },
    {
        "id": "full_trip_003",
        "category": "full_trip",
        "query": "I want to visit Thailand for 7 days, help me plan everything",
        "expected_intent": "full_trip",
        "expected_agents": ["flight_agent", "hotel_agent", "weather_agent", "itinerary_agent"],
    },
    {
        "id": "general_001",
        "category": "general",
        "query": "Tell me about Tokyo",
        "expected_intent": "general",
        "expected_agents": None,  # supervisor should decide — no strict assertion
    },
    {
        "id": "ambiguous_001",
        "category": "general",
        "query": "What should I know before visiting Japan?",
        "expected_intent": "general",
        "expected_agents": None,
    },
    {
        "id": "complex_001",
        "category": "full_trip",
        "query": "Plan a 10-day adventure trip to New Zealand from Singapore, budget under 5000 SGD",
        "expected_intent": "full_trip",
        "expected_agents": ["flight_agent", "hotel_agent", "weather_agent", "itinerary_agent"],
    },
    {
        "id": "multi_city_001",
        "category": "full_trip",
        "query": "Plan a trip visiting both London and Paris over 10 days from New York",
        "expected_intent": "full_trip",
        "expected_agents": ["flight_agent", "hotel_agent", "weather_agent", "itinerary_agent"],
    },
    {
        "id": "follow_up_001",
        "category": "follow_up",
        "query": "Plan a 4-day trip to Kyoto from Singapore next month, mid-range budget",
        "follow_up": "Can you make it cheaper? Suggest budget hotels instead.",
        "expected_intent": "full_trip",
        "expected_agents": ["flight_agent", "hotel_agent", "weather_agent", "itinerary_agent"],
    },
    {
        "id": "non_english_001",
        "category": "full_trip",
        "query": "帮我规划一个从上海出发去东京的5天行程，预算中等",
        "expected_intent": "full_trip",
        "expected_agents": ["flight_agent", "hotel_agent", "weather_agent", "itinerary_agent"],
    },
]
