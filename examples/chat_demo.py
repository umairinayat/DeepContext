"""
DeepContext Example - Chat with hierarchical memory.

This example shows how to use DeepContext as a library
to build a chatbot with persistent, intelligent memory.

Prerequisites:
    pip install deepcontext
    export DEEPCONTEXT_OPENAI_API_KEY="sk-..."
    export DEEPCONTEXT_DATABASE_URL="postgresql+asyncpg://user:pass@localhost/deepcontext"
"""

import asyncio

from openai import AsyncOpenAI

from deepcontext import DeepContext


async def main() -> None:
    # 1. Initialize DeepContext
    ctx = DeepContext(
        # Reads from environment variables if not passed:
        # DEEPCONTEXT_DATABASE_URL, DEEPCONTEXT_OPENAI_API_KEY
    )
    await ctx.init()

    # 2. Initialize chat client (reuse the API key from DeepContext settings)
    chat_client = AsyncOpenAI(api_key=ctx._settings.llm_api_key)
    user_id = "demo_user"
    conversation_id = "demo_conv_1"

    print("=" * 50)
    print("DeepContext Chat Demo")
    print("=" * 50)
    print("Type 'exit' to quit")
    print("Type 'memories' to search your memories")
    print("Type 'graph <entity>' to explore the knowledge graph")
    print("Type 'lifecycle' to run memory maintenance")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        # --- Special commands ---

        if user_input.lower() == "memories":
            query = input("Search query: ").strip()
            results = await ctx.search(query, user_id=user_id, limit=10)
            print(f"\n--- {results.total} memories found ---")
            for r in results.results:
                print(f"  [{r.tier.value}|{r.memory_type.value}] {r.text} (score: {r.score})")
            continue

        if user_input.lower().startswith("graph "):
            entity = user_input[6:].strip()
            neighbors = await ctx.get_entity_graph(user_id, entity, depth=2)
            print(f"\n--- Graph for '{entity}' ---")
            for n in neighbors:
                print(f"  --[{n['relation']}]--> {n['entity']} ({n['entity_type']})")
            if not neighbors:
                print("  No connections found.")
            continue

        if user_input.lower() == "lifecycle":
            result = await ctx.run_lifecycle(user_id)
            print(f"\n--- Lifecycle ---")
            print(f"  Decayed: {result['memories_decayed']}")
            print(f"  Consolidated: {result['memories_consolidated']}")
            print(f"  Cleaned: {result['memories_cleaned']}")
            continue

        # --- Normal chat ---

        # Search for relevant memories
        memory_results = await ctx.search(user_input, user_id=user_id, limit=5)
        memories_str = "\n".join(
            f"- [{r.memory_type.value}] {r.text}" for r in memory_results.results
        )

        # Generate response with memory context
        system_prompt = (
            "You are a helpful AI assistant with persistent memory.\n"
            "Use the user's memories to personalize your responses.\n\n"
            f"User Memories:\n{memories_str or 'No memories yet.'}"
        )

        response = await chat_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        )

        assistant_response = response.choices[0].message.content or ""
        print(f"\nAI: {assistant_response}")

        # Store memories from this turn
        add_result = await ctx.add(
            messages=[
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_response},
            ],
            user_id=user_id,
            conversation_id=conversation_id,
        )

        if add_result.memories_added > 0:
            print(f"  [+{add_result.memories_added} memories stored]")

    # Cleanup
    await ctx.close()
    print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
