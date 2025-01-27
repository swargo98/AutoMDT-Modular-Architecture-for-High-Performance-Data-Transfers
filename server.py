import sys
import zmq
import threading
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.zmq import ZmqServerTransport
from tinyrpc.server import RPCServer
from tinyrpc.dispatch import RPCDispatcher

# Global call counter
call_count = 0

def start_server():
    ctx = zmq.Context()
    dispatcher = RPCDispatcher()
    transport = ZmqServerTransport.create(ctx, 'tcp://127.0.0.1:5001')

    rpc_server = RPCServer(
        transport,
        JSONRPCProtocol(),
        dispatcher
    )

    @dispatcher.public
    def numSort(nums):
        """Sort the list; shut down after receiving 2 calls."""
        global call_count
        call_count += 1
        print(f"numSort called, current count: {call_count}")

        # Once we've called it twice, shut down:
        if call_count >= 2:
            print("numSort called twice. Shutting down the server...")
            sys.exit(0)  # This kills the entire process.

        return sorted(nums)

    print("Server is starting...")
    rpc_server.serve_forever()  # Blocking call.

# Create a background thread to run the server. 
server_thread = threading.Thread(target=start_server, daemon=False)
server_thread.start()

print("Server started in another thread.")

# Keep the main thread alive until server_thread ends. 
# (It will end after 2 calls to numSort, because of sys.exit(0).)
server_thread.join()

print("Main thread exiting.")
