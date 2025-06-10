import cv2
import numpy as np
import speech_recognition as sr
from googletrans import Translator, LANGUAGES
from gtts import gTTS
import pygame
import io
import threading
import queue
import socket
import pickle
import struct
import select
import time

class VideoCallTranslator:
    def __init__(self):
        # Initialize components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.translator = Translator()
        pygame.mixer.init()
        
        # Queues for inter-thread communication
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.translation_queue = queue.Queue()
        self.video_queue = queue.Queue()
        self.network_queue = queue.Queue()
        
        # Configuration with Hindi added
        self.languages = {
            'English': 'en',
            'Hindi': 'hi',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Japanese': 'ja',
            'Chinese': 'zh-CN',
            'Russian': 'ru',
            'Arabic': 'ar'
        }
        self.source_lang = 'en'
        self.target_lang = 'hi'  # Default to Hindi as target
        self.running = False
        self.translating = True
        self.in_call = False
        self.is_host = False
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video device")
        self.frame_size = (640, 480)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Networking
        self.socket = None
        self.connection = None
        self.host = '0.0.0.0'
        self.port = 5000
        self.remote_frame = None
        
        # Audio setup
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def show_language_menu(self):
        """Display language selection menu with Hindi option"""
        print("\nAvailable Languages:")
        for i, (name, code) in enumerate(self.languages.items(), 1):
            print(f"{i}. {name} ({code})")
        
        while True:
            try:
                src_choice = int(input("\nSelect your language (number): "))
                tgt_choice = int(input("Select target language (number): "))
                
                if 1 <= src_choice <= len(self.languages) and 1 <= tgt_choice <= len(self.languages):
                    self.source_lang = list(self.languages.values())[src_choice-1]
                    self.target_lang = list(self.languages.values())[tgt_choice-1]
                    print(f"\nTranslation set: {list(self.languages.keys())[src_choice-1]} -> {list(self.languages.keys())[tgt_choice-1]}")
                    break
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a number.")

    def setup_call(self):
        """Configure call settings"""
        print("\nCall Options:")
        print("1. Start a new call (host)")
        print("2. Join existing call")
        print("3. Local demo (no call)")
        
        while True:
            choice = input("Select option (1-3): ")
            if choice == '1':
                self.is_host = True
                self.port = int(input("Enter port to host on (default 5000): ") or "5000")
                self.in_call = True
                break
            elif choice == '2':
                self.is_host = False
                self.host = input("Enter host IP (default localhost): ") or "localhost"
                self.port = int(input("Enter port (default 5000): ") or "5000")
                self.in_call = True
                break
            elif choice == '3':
                self.in_call = False
                break
            else:
                print("Invalid choice")

    def audio_capture_thread(self):
        """Capture audio from microphone"""
        with self.microphone as source:
            while self.running:
                try:
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=5)
                    self.audio_queue.put(audio)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Audio capture error: {e}")
                    time.sleep(0.1)

    def speech_recognition_thread(self):
        """Convert audio to text"""
        while self.running:
            if not self.audio_queue.empty():
                audio = self.audio_queue.get()
                try:
                    text = self.recognizer.recognize_google(audio, language=self.source_lang)
                    self.text_queue.put(text)
                    if self.in_call:
                        self.network_queue.put(('text', text))
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                except Exception as e:
                    print(f"Recognition error: {e}")

    def translation_thread(self):
        """Translate text to target language (including Hindi)"""
        while self.running:
            if not self.text_queue.empty():
                text = self.text_queue.get()
                if self.translating:
                    try:
                        translation = self.translator.translate(text, src=self.source_lang, dest=self.target_lang).text
                        self.translation_queue.put(translation)
                        
                        # Convert translation to speech
                        tts = gTTS(translation, lang=self.target_lang)
                        fp = io.BytesIO()
                        tts.write_to_fp(fp)
                        fp.seek(0)
                        
                        # Play audio
                        pygame.mixer.music.load(fp)
                        pygame.mixer.music.play()
                    except Exception as e:
                        print(f"Translation error: {e}")

    def video_capture_thread(self):
        """Capture and process video frames"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue
            
            frame = cv2.resize(frame, self.frame_size)
            
            # Display original text if available
            if not self.text_queue.empty():
                text = self.text_queue.queue[-1]
                cv2.putText(frame, f"Original: {text}", (10, 30), self.font, 0.7, (0, 255, 0), 2)
            
            # Display translation if available
            if not self.translation_queue.empty() and self.translating:
                translation = self.translation_queue.queue[-1]
                cv2.putText(frame, f"Translated: {translation}", (10, 70), self.font, 0.7, (0, 0, 255), 2)
            
            # Show local video
            cv2.imshow('Video Translator', frame)
            
            # Send frame if in call
            if self.in_call:
                self.network_queue.put(('video', frame))
            
            # Show remote video if available
            if self.remote_frame is not None:
                cv2.imshow('Remote Video', self.remote_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

    def network_thread(self):
        """Handle network communication with proper framing"""
        if not self.in_call:
            return
            
        try:
            if self.is_host:
                # Host mode
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.socket.bind((self.host, self.port))
                self.socket.listen(1)
                print(f"Waiting for connection on {self.host}:{self.port}...")
                self.connection, addr = self.socket.accept()
                print(f"Connected to {addr}")
            else:
                # Client mode
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                print(f"Connecting to {self.host}:{self.port}...")
                self.socket.connect((self.host, self.port))
                self.connection = self.socket
                print("Connected to host")
            
            self.connection.settimeout(0.1)
            
            while self.running and self.connection:
                # Send data
                try:
                    if not self.network_queue.empty():
                        data_type, data = self.network_queue.get()
                        
                        if data_type == 'video':
                            # Serialize frame
                            data = pickle.dumps(data)
                            # Send message type (1 byte) and size (8 bytes)
                            header = struct.pack("!BQ", 0, len(data))  # 0 for video
                            self.connection.sendall(header + data)
                        elif data_type == 'text':
                            # Send text with header
                            data = data.encode('utf-8')
                            header = struct.pack("!BQ", 1, len(data))  # 1 for text
                            self.connection.sendall(header + data)
                except (ConnectionResetError, BrokenPipeError, socket.timeout):
                    continue
                except Exception as e:
                    print(f"Send error: {e}")
                    break
                
                # Receive data
                try:
                    # Receive header (1 byte type + 8 byte size)
                    header = self.connection.recv(9)
                    if not header:
                        continue
                    
                    # Unpack header
                    msg_type, msg_size = struct.unpack("!BQ", header)
                    
                    # Receive actual data
                    received = 0
                    chunks = []
                    while received < msg_size:
                        chunk = self.connection.recv(min(msg_size - received, 4096))
                        if not chunk:
                            break
                        chunks.append(chunk)
                        received += len(chunk)
                    
                    if received < msg_size:
                        continue  # Incomplete message, discard
                    
                    data = b''.join(chunks)
                    
                    if msg_type == 0:  # Video frame
                        try:
                            frame = pickle.loads(data)
                            self.remote_frame = frame
                        except Exception as e:
                            print(f"Frame unpickling error: {e}")
                    elif msg_type == 1:  # Text data
                        try:
                            text = data.decode('utf-8')
                            print(f"Received text: {text}")
                        except Exception as e:
                            print(f"Text decode error: {e}")
                            
                except socket.timeout:
                    continue
                except struct.error:
                    continue
                except Exception as e:
                    print(f"Receive error: {e}")
                    break
                    
        except Exception as e:
            print(f"Network setup error: {e}")
        finally:
            if self.connection:
                self.connection.close()
            if self.socket:
                self.socket.close()
            print("Network connection closed")

    def start(self):
        """Start all components"""
        self.show_language_menu()
        self.setup_call()
        
        self.running = True
        
        # Create threads
        threads = [
            threading.Thread(target=self.audio_capture_thread),
            threading.Thread(target=self.speech_recognition_thread),
            threading.Thread(target=self.translation_thread),
            threading.Thread(target=self.video_capture_thread),
            threading.Thread(target=self.network_thread) 
        ]
        
        # Start threads
        for t in threads:
            t.daemon = True
            t.start()
        
        # Wait for video thread to finish (it will set running=False when window is closed)
        threads[3].join()
        
        # Cleanup
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        if self.connection:
            self.connection.close()
        if self.socket:
            self.socket.close()
        print("Program shutdown complete")

if __name__ == "__main__":
    try:
        print("=== Real-Time Video Call Translator ===")
        translator = VideoCallTranslator()
        translator.start()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Program ended")