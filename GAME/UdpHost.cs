using System;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Collections.Generic;
using UnityEngine;

namespace Udp
{
    public class UdpHost : MonoBehaviour
    {
        public static event Action<string> OnReceiveMsg;
        public static event Action<string> OnClientError;

        [SerializeField, Tooltip("Set to true if you want the debug logs to show in the console")] 
        protected bool _consoleLogsEnabled = true;

        [Header("Host settings")]
        [SerializeField] protected Int32 _hostPort = 5013;
        [SerializeField] protected string _hostIp = "0.0.0.0";

        [Header("Client settings")]
        [SerializeField] protected Int32 _clientPort = 5011;
        [SerializeField] protected string _clientIp = "127.0.0.1";

        [SerializeField, Tooltip("Set true if host should auto start and connect to the client")]
        protected bool _autoConnect = true;

        [Header("Stream")]
        [SerializeField] public string _message;

        protected Thread _socketThread = null;
        protected bool _connected;
        protected EndPoint _client;
        protected Socket _socket;

        // 🧠 Thread-safe queue for main thread message handling
        private readonly Queue<string> _mainThreadMessageQueue = new Queue<string>();
        private readonly object _queueLock = new object();

        public virtual void Start()
        {
            if (_autoConnect) Connect();
        }

        public virtual void Connect()
        {
            if (IsClientValid() && IsHostValid())
            {
                _socketThread = new Thread(ExecuteHost);
                _socketThread.IsBackground = true;
                _socketThread.Start();
            }
        }

        public virtual void Close()
        {
            _connected = false;
        }

        void Update()
        {
            // ✅ Process messages safely on main thread
            lock (_queueLock)
            {
                while (_mainThreadMessageQueue.Count > 0)
                {
                    string msg = _mainThreadMessageQueue.Dequeue();
                    _message = msg;
                    OnReceiveMsg?.Invoke(msg);
                }
            }
        }

        public virtual void SetClient(string clientIp, Int32 clientPort)
        {
            _clientIp = clientIp;
            _clientPort = clientPort;
        }

        public virtual void SetHost(string hostIp, Int32 hostPort)
        {
            _hostIp = hostIp;
            _hostPort = hostPort;
        }

        public virtual void SendMsg(string msg)
        {
            if (_connected)
            {
                byte[] data = Encoding.ASCII.GetBytes(msg);
                _socket.SendTo(data, data.Length, SocketFlags.None, _client);
            }
            else
            {
                Log("Not connected, can't send a message");
            }
        }

        public virtual void MessageReceived(string message)
        {
            lock (_queueLock)
            {
                _mainThreadMessageQueue.Enqueue(message);
            }
        }

        protected virtual void ExecuteHost()
        {
            try
            {
                int recv;
                byte[] data = new byte[1024];
                IPEndPoint ipep = new IPEndPoint(IPAddress.Parse(_hostIp), _hostPort);

                _socket = new Socket(AddressFamily.InterNetwork,
                                SocketType.Dgram, ProtocolType.Udp);

                _socket.Bind(ipep);
                Log("Waiting for a client...");

                try
                {
                    EndPoint remote = new IPEndPoint(IPAddress.Any, 0);
                    Log("Waiting for first message from client...");

                    recv = _socket.ReceiveFrom(data, ref remote);
                    string msg = Encoding.ASCII.GetString(data, 0, recv);
                    Debug.Log("mess recived: " + msg);
                    MessageReceived(msg);

                    _client = remote;
                    _connected = true;
                    Log(" Client connected to: " + remote.ToString());

                    while (_connected)
                    {
                        data = new byte[1024];
                        recv = _socket.ReceiveFrom(data, ref _client);

                        string receivedMsg = Encoding.ASCII.GetString(data, 0, recv);
                        Debug.Log("✅ Mess recived: " + receivedMsg);
                        MessageReceived(receivedMsg);
                    }
                }
                catch (Exception e)
                {
                    if (e is ThreadAbortException)
                    {
                        Log("UDP thread terminated on exit.");
                    }
                    else
                    {
                        Debug.LogError("Err in communication to UDP: " + e.Message);
                    }
                }

            }
            catch (SocketException e)
            {
                if (e.ErrorCode == 10051)
                {
                    string msg = "Client is unreachable, check the IP or port " + e.ErrorCode;
                    OnClientError?.Invoke(msg);
                    Log(msg);
                }
                else if (e.ErrorCode == 10054)
                {
                    OnClientError?.Invoke(e.Message);
                    Log(e.Message);
                }
                else if (e.ErrorCode == 10048)
                {
                    string msg = "The host address and port is already in use!";
                    Log(msg);
                }
                else
                {
                    Log(e.Message + " errorcode:" + e.ErrorCode);
                }
            }
            finally
            {
                Log("Closing...");
                if (_socket.Connected) _socket.Shutdown(SocketShutdown.Both);
                _socket.Close();
                _connected = _socket.Connected;
            }
        }

        protected bool IsClientValid()
        {
            if (!IsIpValid(_clientIp))
            {
                Log("Client IP is not a valid IP!");
                return false;
            }
            return true;
        }

        protected bool IsHostValid()
        {
            if (!IsIpValid(_hostIp))
            {
                Log("Host IP is not a valid IP!");
                return false;
            }
            return true;
        }

        protected bool IsIpValid(string ip)
        {
            string ipPattern = @"^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$";
            Regex regex = new Regex(ipPattern);
            return !string.IsNullOrEmpty(ip) && regex.IsMatch(ip, 0);
        }

        protected void Log(string msg)
        {
            if (_consoleLogsEnabled)
                Debug.Log($"UDP HOST: {msg}");
        }

        public virtual void OnApplicationQuit()
        {
            Close();
        }
    }
}
