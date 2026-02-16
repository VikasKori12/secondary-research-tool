import asyncio
from research_system.agent import run_web_research

async def main():
    result = await run_web_research("What are recent developments in renewable hydrogen?")
    print(type(result))
    try:
        print(result)
    except Exception:
        pass

if __name__ == "__main__":
    asyncio.run(main())
