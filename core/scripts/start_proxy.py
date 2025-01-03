import socket
import threading

# 配置
LISTEN_HOST = '127.0.0.1'  # 监听的 IP
LISTEN_PORT = 8090        # 监听的端口
FORWARD_HOST = '10.8.0.2'  # 转发的目标 IP
FORWARD_PORT = 8090         # 转发的目标端口

def handle_client(client_socket):
    """处理客户端连接并转发数据"""
    try:
        # 连接到目标服务器
        forward_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        forward_socket.connect((FORWARD_HOST, FORWARD_PORT))

        # 创建两个线程，一个负责从客户端读取数据并发送到目标服务器，
        # 另一个负责从目标服务器读取数据并发送到客户端。
        def forward_data(source, destination):
            while True:
                try:
                    data = source.recv(4096)
                    if not data:
                        break
                    destination.sendall(data)
                except Exception:
                    break

        # 启动线程
        client_to_server = threading.Thread(target=forward_data, args=(client_socket, forward_socket))
        server_to_client = threading.Thread(target=forward_data, args=(forward_socket, client_socket))
        client_to_server.start()
        server_to_client.start()

        # 等待线程结束
        client_to_server.join()
        server_to_client.join()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()
        forward_socket.close()

def start_proxy():
    """启动代理服务器"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((LISTEN_HOST, LISTEN_PORT))
    server.listen(5)
    print(f"Proxy server listening on {LISTEN_HOST}:{LISTEN_PORT} and forwarding to {FORWARD_HOST}:{FORWARD_PORT}")

    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        # 为每个客户端启动一个线程
        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()

if __name__ == "__main__":
    start_proxy()