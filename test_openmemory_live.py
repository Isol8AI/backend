#!/usr/bin/env python
"""
Test script for OpenMemory Python SDK with live Supabase connection.

This tests the full CRUD flow:
1. Store a memory with pre-computed embedding
2. Search for it by embedding similarity
3. Get it by ID
4. List all memories for the user
5. Delete it
"""
import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the OpenMemory SDK to path
memory_path = Path(__file__).parent.parent / "memory" / "packages" / "openmemory-py" / "src"
sys.path.insert(0, str(memory_path))

async def main():
    print("=" * 60)
    print("OpenMemory SDK Live Supabase Test")
    print("=" * 60)

    # Verify environment
    backend = os.getenv("OM_METADATA_BACKEND")
    dsn = os.getenv("OM_PG_DSN")

    print("\nEnvironment:")
    print(f"  OM_METADATA_BACKEND: {backend}")
    print(f"  OM_PG_DSN: {'***' if dsn else 'NOT SET'}")

    if backend != "postgres":
        print("\n❌ ERROR: OM_METADATA_BACKEND must be 'postgres'")
        return 1

    if not dsn:
        print("\n❌ ERROR: OM_PG_DSN not set")
        return 1

    # Import after env is loaded
    from openmemory import Memory
    from openmemory.core.db import is_pg

    print(f"\n  PostgreSQL mode: {is_pg}")

    if not is_pg:
        print("\n❌ ERROR: OpenMemory not in PostgreSQL mode")
        return 1

    # Initialize Memory
    print("\n1. Initializing Memory SDK...")
    try:
        mem = Memory()
        print("   ✓ Memory initialized")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return 1

    # Test data
    test_user_id = "user_test_live_connection"
    test_content = "Test memory: Claude is an AI assistant made by Anthropic."
    # Simple 384-dim embedding (MiniLM-L6-v2 style)
    test_embedding = [0.1] * 384
    test_sector = "semantic"
    test_tags = ["test", "live_connection"]
    test_metadata = {"test_run": True, "iv": "test_iv", "tag": "test_tag"}

    # 2. Store memory
    print("\n2. Storing test memory...")
    try:
        result = await mem.add_with_embedding(
            content=test_content,
            embedding=test_embedding,
            user_id=test_user_id,
            sector=test_sector,
            tags=test_tags,
            metadata=test_metadata,
        )
        print(f"   ✓ Memory stored: {result}")
        memory_id = result.get("id")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 3. Search memories
    print("\n3. Searching for memory by embedding...")
    try:
        results = await mem.search_with_embedding(
            query_text="Claude AI",
            query_embedding=test_embedding,
            user_id=test_user_id,
            limit=5,
        )
        print(f"   ✓ Found {len(results)} result(s)")
        if results:
            print(f"   First result ID: {results[0].get('id')}")
            print(f"   Score: {results[0].get('score', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 4. Get memory by ID
    print("\n4. Getting memory by ID...")
    try:
        memory = await mem.get(memory_id)
        if memory:
            print(f"   ✓ Got memory: {memory.get('id')}")
            print(f"   Content: {memory.get('content')[:50]}...")
            print(f"   Sector: {memory.get('primary_sector')}")
        else:
            print("   ❌ Memory not found")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 5. List memories (history)
    print("\n5. Listing memories for user...")
    try:
        history = await mem.history(user_id=test_user_id, limit=10, offset=0)
        print(f"   ✓ Found {len(history)} memory(ies)")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 6. Delete memory
    print("\n6. Deleting test memory...")
    try:
        await mem.delete(memory_id)
        print("   ✓ Memory deleted")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 7. Verify deletion
    print("\n7. Verifying deletion...")
    try:
        memory = await mem.get(memory_id)
        if memory is None:
            print("   ✓ Memory successfully deleted")
        else:
            print("   ❌ Memory still exists!")
            return 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return 1

    # Cleanup: Close pool
    print("\n8. Cleaning up...")
    try:
        from openmemory.core.db import close_pg_pool
        await close_pg_pool()
        print("   ✓ Connection pool closed")
    except Exception as e:
        print(f"   ⚠ Cleanup warning: {e}")

    print("\n" + "=" * 60)
    print("✓ All tests passed! OpenMemory SDK works with Supabase.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
