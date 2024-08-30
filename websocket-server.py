import asyncio
import websockets
import json

clients = set()

async def handler(websocket):
    clients.add(websocket)
    try:
        async for message in websocket:
            print("message: ", message)
            pass  
    finally:
        clients.remove(websocket)

async def broadcast(data):
    if clients:
        message = json.dumps(data)
        await asyncio.wait([client.send(message) for client in clients])

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Start server")
        await asyncio.Future()  

if __name__ == "__main__":
    asyncio.run(main())
