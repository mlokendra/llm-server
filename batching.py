import asyncio
from medusa_runner import medusa_generate

request_queue = asyncio.Queue()
response_futures = []

async def queue_request(prompt):
    future = asyncio.Future()
    await request_queue.put((prompt, future))
    return await future

async def batch_worker():
    while True:
        if request_queue.empty():
            await asyncio.sleep(0.01)
            continue
        batch = []
        futures = []
        while not request_queue.empty():
            prompt, fut = await request_queue.get()
            batch.append(prompt)
            futures.append(fut)
        results = medusa_generate(batch)
        for fut, res in zip(futures, results):
            fut.set_result(res)


# Start batch worker
async def main():
    task = asyncio.create_task(batch_worker())
    await task  # or await asyncio.sleep(...)

asyncio.run(main())