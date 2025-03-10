import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import whisper
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import queue
import time
from datetime import datetime

# Suppress FP16 warning for CPU usage
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Ensure VADER lexicon is available
nltk.download('vader_lexicon', quiet=True)

# Initialize VADER SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# ========== Step 1: Real-time Audio Transcription using Whisper ==========

class WhisperTranscriber:
    def __init__(self, callback=None):
        self.model = None
        self.callback = callback
        self.stop_flag = threading.Event()
        self.text_queue = queue.Queue()
        self.current_text = ""
        # Track if model is currently loading to prevent duplicate loads
        self.is_loading = False
        # Add completion flag to track when transcription is done
        self.transcription_complete = False

    def load_model(self):
        """Load the Whisper model."""
        if self.model is None and not self.is_loading:
            self.is_loading = True
            # Update status while loading
            if hasattr(self, 'status_callback') and self.status_callback:
                self.status_callback("Loading Whisper model... Please wait.")
            self.model = whisper.load_model("medium")
            self.is_loading = False
            # Update status after loading
            if hasattr(self, 'status_callback') and self.status_callback:
                self.status_callback("Model loaded. Processing audio...")
        return self.model

    def set_status_callback(self, callback):
        """Set a callback for status updates."""
        self.status_callback = callback

    def transcribe_chunk(self, audio_path, start_time=0, chunk_duration=10):
        """Transcribe a chunk of audio."""
        try:
            # Load model if not already loaded
            model = self.load_model()
            
            # Transcribe the chunk
            result = model.transcribe(
                audio_path,
                initial_prompt=self.current_text,  # Use previous text as context
                condition_on_previous_text=True
            )
            
            # Get the transcribed text
            return result["text"]
        except Exception as e:
            print(f"Error during chunk transcription: {e}")
            return None

    def transcribe_realtime(self, audio_path):
        """Simulate real-time transcription by processing chunks of audio."""
        self.stop_flag.clear()
        self.current_text = ""
        self.transcription_complete = False
        
        # Process just one complete audio file instead of chunks
        try:
            # Update status
            if hasattr(self, 'status_callback') and self.status_callback:
                self.status_callback("Processing audio file... Please wait.")
            
            # Transcribe the entire audio file at once
            chunk_text = self.transcribe_chunk(audio_path)
            
            if chunk_text:
                # Update current text
                self.current_text = chunk_text
                
                # Put in queue for UI thread to consume
                self.text_queue.put(chunk_text)
                
                # Call callback if provided
                if self.callback:
                    self.callback(chunk_text)
        except Exception as e:
            print(f"Error during transcription: {e}")
        
        # Signal completion
        self.transcription_complete = True
        self.text_queue.put(None)

    def stop_transcription(self):
        """Stop the ongoing transcription."""
        self.stop_flag.set()

# ========== Step 2: Analyze Sentiment using VADER ==========

def analyze_sentiment(text):
    """Analyze sentiment of a given text using VADER."""
    sentiment_score = sia.polarity_scores(text)
    compound = sentiment_score["compound"]

    # Determine sentiment
    if compound >= 0.05:
        sentiment = "Positive 😊"
    elif compound <= -0.05:
        sentiment = "Negative 😠"
    else:
        sentiment = "Neutral 😐"

    return sentiment, compound

# ========== Step 3: Real-time Conversation Analysis ==========

class ConversationAnalyzer:
    def __init__(self):
        self.conversation = []  # Store as sequential conversation
        self.current_speaker = "customer"  # Start with customer
        self.buffer = ""
        self.sentence_buffer = []
        self.customer_scores = []
        self.bot_scores = []
        self.message_counter = {"customer": 0, "bot": 0}
    
    def process_text(self, text):
        """Process incoming text and assign to speakers."""
        # Add text to buffer
        self.buffer += text
        
        # Split buffer into sentences (simple split by period)
        sentences = self.buffer.split(". ")
        
        # Last sentence might be incomplete, keep it in buffer
        if len(sentences) > 1:
            self.buffer = sentences.pop()
            
            # Process complete sentences
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:  # Ignore empty sentences
                    self.sentence_buffer.append(sentence + ".")
        
        # Process sentences in buffer
        self._process_sentence_buffer()
        
        return self.conversation
    
    def _process_sentence_buffer(self):
        """Process sentences in buffer and assign to speakers alternately."""
        while self.sentence_buffer:
            sentence = self.sentence_buffer.pop(0)
            if sentence:
                # Analyze sentiment
                sentiment, score = analyze_sentiment(sentence)
                
                # Increment message counter for current speaker
                self.message_counter[self.current_speaker] += 1
                idx = self.message_counter[self.current_speaker]
                
                # Store sentiment score
                if self.current_speaker == "customer":
                    self.customer_scores.append((idx, score))
                else:
                    self.bot_scores.append((idx, score))
                
                # Add to conversation with speaker info
                self.conversation.append({
                    "speaker": self.current_speaker,
                    "text": sentence,
                    "sentiment": sentiment,
                    "score": score,
                    "message_index": idx
                })
                
                # Switch speaker for the next sentence
                self.current_speaker = "bot" if self.current_speaker == "customer" else "customer"
    
    def finalize(self):
        """Process any remaining text in the buffer."""
        if self.buffer:
            self.sentence_buffer.append(self.buffer + ".")
            self.buffer = ""
            self._process_sentence_buffer()

# ========== GUI Functions ==========

def select_audio_file():
    """Open a file dialog to select an audio file."""
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        audio_file.set(file_path)
        analyze_button.config(state=tk.NORMAL)
        update_status("Audio file selected. Click 'Analyze' to process.")

def update_status(message):
    """Update status bar message."""
    status_var.set(message)
    root.update_idletasks()

def analyze_audio_threaded():
    """Start analysis in a separate thread to keep GUI responsive."""
    threading.Thread(target=analyze_audio, daemon=True).start()

def update_ui():
    """Update UI from the transcription queue."""
    global transcriber, conversation_analyzer, ui_update_running
    
    if ui_update_running:
        try:
            # Check if there's new text in the queue
            while not transcriber.text_queue.empty():
                chunk_text = transcriber.text_queue.get_nowait()
                
                # None signals end of transcription
                if chunk_text is None:
                    # Finalize conversation analysis
                    conversation_analyzer.finalize()
                    
                    # Update UI to show completion
                    progress_bar.stop()
                    analyze_button.config(state=tk.NORMAL)
                    update_status("Analysis complete!")
                    
                    # Final update of displays
                    update_transcript_display()
                    update_sentiment_display()
                    update_sentiment_graph(conversation_analyzer.customer_scores, 
                                         conversation_analyzer.bot_scores)
                    
                    # Stop UI update loop
                    ui_update_running = False
                    return
                
                # Update transcript display with the new chunk
                transcript_text.config(state=tk.NORMAL)
                transcript_text.insert(tk.END, chunk_text + " ")
                transcript_text.see(tk.END)
                transcript_text.config(state=tk.DISABLED)
                
                # Process text for conversation analysis
                conversation_analyzer.process_text(chunk_text)
                
                # Update displays
                update_sentiment_display()
                update_sentiment_graph(conversation_analyzer.customer_scores, 
                                     conversation_analyzer.bot_scores)
        except queue.Empty:
            pass
        
        # Schedule the next update only if we're still running
        if ui_update_running:
            root.after(100, update_ui)

def update_transcript_display():
    """Update the full transcript display."""
    transcript_text.config(state=tk.NORMAL)
    transcript_text.delete(1.0, tk.END)
    
    full_transcript = ""
    for msg in conversation_analyzer.conversation:
        full_transcript += msg["text"] + " "
    
    transcript_text.insert(tk.END, full_transcript)
    transcript_text.see(tk.END)
    transcript_text.config(state=tk.DISABLED)

def update_sentiment_display():
    """Update the sentiment analysis display as a conversation."""
    sentiment_text.config(state=tk.NORMAL)
    sentiment_text.delete(1.0, tk.END)
    
    # Display as back-and-forth conversation
    for msg in conversation_analyzer.conversation:
        speaker = msg["speaker"].capitalize()
        text = msg["text"]
        sentiment = msg["sentiment"]
        score = msg["score"]
        
        # Apply tag based on speaker
        if msg["speaker"] == "customer":
            sentiment_text.insert(tk.END, f"Customer: ", "customer")
        else:
            sentiment_text.insert(tk.END, f"Bot: ", "bot")
        
        sentiment_text.insert(tk.END, f"{text}\n")
        
        # Apply tag based on sentiment
        if score >= 0.05:
            sentiment_text.insert(tk.END, f"[{sentiment} (Score: {score:.2f})]\n\n", "positive")
        elif score <= -0.05:
            sentiment_text.insert(tk.END, f"[{sentiment} (Score: {score:.2f})]\n\n", "negative")
        else:
            sentiment_text.insert(tk.END, f"[{sentiment} (Score: {score:.2f})]\n\n", "neutral")
    
    sentiment_text.see(tk.END)
    sentiment_text.config(state=tk.DISABLED)

def analyze_audio():
    """Start real-time transcription and analysis of the audio."""
    global transcriber, conversation_analyzer, ui_update_running
    
    file_path = audio_file.get()
    if not file_path:
        return

    # Create a new conversation analyzer
    conversation_analyzer = ConversationAnalyzer()
    
    # Update UI to show processing
    analyze_button.config(state=tk.DISABLED)
    progress_bar.start(10)
    update_status("Initializing transcription... Please wait.")
    
    # Clear previous transcript and sentiment analysis
    transcript_text.config(state=tk.NORMAL)
    transcript_text.delete(1.0, tk.END)
    transcript_text.config(state=tk.DISABLED)
    
    sentiment_text.config(state=tk.NORMAL)
    sentiment_text.delete(1.0, tk.END)
    sentiment_text.config(state=tk.DISABLED)
    
    # Create and start transcriber
    transcriber = WhisperTranscriber()
    transcriber.set_status_callback(update_status)
    
    # Start transcription in a separate thread
    threading.Thread(
        target=transcriber.transcribe_realtime,
        args=(file_path,),
        daemon=True
    ).start()
    
    # Start UI update scheduler
    ui_update_running = True
    root.after(100, update_ui)

def update_sentiment_graph(customer_scores, bot_scores):
    """Update the sentiment graph with customer and bot scores."""
    ax.clear()

    # Extract message indices and scores
    customer_indices = [idx for idx, _ in customer_scores]
    customer_sentiments = [score for _, score in customer_scores]
    bot_indices = [idx for idx, _ in bot_scores]
    bot_sentiments = [score for _, score in bot_scores]

    # Plot customer and bot sentiment scores
    if customer_sentiments:
        ax.plot(customer_indices, customer_sentiments, marker='o', color='#2ecc71', label='Customer', 
                linestyle='-', linewidth=2, markersize=8)
    
    if bot_sentiments:
        ax.plot(bot_indices, bot_sentiments, marker='s', color='#9b59b6', label='Bot', 
                linestyle='-', linewidth=2, markersize=8)

    # Add a horizontal line at y=0 for reference
    ax.axhline(y=0, color='#95a5a6', linestyle='--', alpha=0.7)
    
    # Add color bands for sentiment regions
    ax.axhspan(0.05, 1.0, alpha=0.2, color='#2ecc71')  # Positive - green
    ax.axhspan(-0.05, 0.05, alpha=0.1, color='#bdc3c7')  # Neutral - gray
    ax.axhspan(-1.0, -0.05, alpha=0.2, color='#e74c3c')  # Negative - red

    # Set labels and title
    ax.set_xlabel("Message Number", fontsize=12, fontweight='bold')
    ax.set_ylabel("Sentiment Score", fontsize=12, fontweight='bold')
    ax.set_title("Real-time Conversation Sentiment", fontsize=14, fontweight='bold')
    
    # Set y-axis limits with some padding
    ax.set_ylim(-1.1, 1.1)
    
    # Set x-axis to be integers only
    max_messages = max(max(customer_indices) if customer_indices else 0, 
                     max(bot_indices) if bot_indices else 0, 1)
    ax.set_xticks(range(1, max_messages + 1))
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend with custom styling
    legend = ax.legend(fontsize=10, loc='lower right', frameon=True, framealpha=0.9)
    frame = legend.get_frame()
    frame.set_facecolor('#f8f9fa')
    frame.set_edgecolor('#e9ecef')

    # Draw the canvas
    canvas.draw()

# ========== Create GUI Widgets ==========
def create_gui():
    global root, audio_file, analyze_button, transcript_text, sentiment_text
    global fig, ax, canvas, progress_bar, status_var
    
    # Create the main window with custom styling
    root = tk.Tk()
    root.title("QA-BOT: Voice Agent Performance Analyzer")
    root.geometry("1280x800")
    root.configure(bg="#1e272e")  # Dark blue-gray background
    
    # Set app icon if available
    try:
        root.iconbitmap("qa_bot_icon.ico")  # Replace with your icon if available
    except:
        pass
    
    # Variable to store the audio file path
    audio_file = tk.StringVar()
    status_var = tk.StringVar()
    status_var.set("Select an audio file to begin analysis.")
    
    # Create custom styles with modern color scheme
    style = ttk.Style()
    
    # Configure the theme
    style.theme_use('clam')
    
    # Modern color palette
    colors = {
        'bg_dark': '#1e272e',      # Dark blue-gray
        'bg_medium': '#2d3436',    # Medium blue-gray
        'bg_light': '#353b48',     # Light blue-gray
        'accent1': '#0984e3',      # Vibrant blue
        'accent2': '#00cec9',      # Teal
        'accent3': '#6c5ce7',      # Purple
        'text_light': '#dfe6e9',   # Light gray
        'text_dark': '#2d3436',    # Dark gray
        'positive': '#00b894',     # Green
        'neutral': '#636e72',      # Gray
        'negative': '#d63031',     # Red
        'customer': '#00a8ff',     # Customer color
        'bot': '#9c88ff'           # Bot color
    }
    
    # Configure frame styles
    style.configure("TFrame", background=colors['bg_dark'])
    style.configure("TLabelframe", background=colors['bg_dark'], bordercolor=colors['bg_medium'])
    style.configure("TLabelframe.Label", font=("Segoe UI", 11, "bold"), 
                   background=colors['bg_dark'], foreground=colors['text_light'])
    
    # Configure label styles
    style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), 
                   background=colors['bg_dark'], foreground=colors['text_light'])
    style.configure("Status.TLabel", font=("Segoe UI", 10), 
                   background=colors['bg_dark'], foreground=colors['accent2'])
    
    # Button styles with gradient-like effect
    style.configure("TButton", font=("Segoe UI", 10, "bold"), 
                   background=colors['accent1'], foreground=colors['text_light'])
    style.map("TButton", 
              background=[("active", colors['accent2']), ("disabled", colors['bg_medium'])],
              foreground=[("disabled", colors['neutral'])])
    
    style.configure("Browse.TButton", font=("Segoe UI", 10), 
                   background=colors['accent3'], foreground=colors['text_light'])
    style.map("Browse.TButton", background=[("active", colors['accent2'])])
    
    style.configure("Analyze.TButton", font=("Segoe UI", 10, "bold"), 
                   background=colors['accent1'], foreground=colors['text_light'])
    style.map("Analyze.TButton", background=[("active", colors['accent2']), 
                                           ("disabled", colors['bg_medium'])])
    
    # Progress bar with gradient colors
    style.configure("Horizontal.TProgressbar", 
                  background=colors['accent2'], 
                  troughcolor=colors['bg_medium'],
                  bordercolor=colors['bg_light'],
                  lightcolor=colors['accent1'],
                  darkcolor=colors['accent3'])
    
    # Create main container
    main_container = ttk.Frame(root)
    main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
    
    # Create header with logo and title
    header_frame = ttk.Frame(main_container)
    header_frame.pack(fill=tk.X, pady=(0, 15))
    
    # Add current date/time to header
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_label = ttk.Label(header_frame, 
                         text=f"Session: {current_datetime}", 
                         style="Status.TLabel")
    date_label.pack(side=tk.RIGHT)
    
    app_title = ttk.Label(header_frame, 
                        text="🤖 QA-BOT: Voice Agent Performance Analyzer", 
                        style="Header.TLabel")
    app_title.pack(side=tk.LEFT)
    
    # File Selection Section with modern styling
    file_frame = ttk.LabelFrame(main_container, text="Audio Source")
    file_frame.pack(fill=tk.X, pady=(0, 15))
    
    file_entry = ttk.Entry(file_frame, textvariable=audio_file, width=70, 
                         font=("Segoe UI", 10))
    file_entry.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)
    
    browse_button = ttk.Button(file_frame, text="Browse", command=select_audio_file, 
                             style="Browse.TButton", width=10)
    browse_button.pack(side=tk.LEFT, padx=10, pady=10)
    
    analyze_button = ttk.Button(file_frame, text="▶ Analyze", command=analyze_audio_threaded, 
                              style="Analyze.TButton", width=10, state=tk.DISABLED)
    analyze_button.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Status bar and progress with modern styling
    status_frame = ttk.Frame(main_container)
    status_frame.pack(fill=tk.X, pady=(0, 15))
    
    status_label = ttk.Label(status_frame, textvariable=status_var, style="Status.TLabel")
    status_label.pack(side=tk.LEFT, padx=5)
    
    progress_bar = ttk.Progressbar(status_frame, mode="indeterminate", 
                                 style="Horizontal.TProgressbar")
    progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
    
    # Content area with Panedwindow
    paned_window = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
    paned_window.pack(fill=tk.BOTH, expand=True)
    
    # Left Panel (Transcript) with modern styling
    left_frame = ttk.Frame(paned_window)
    paned_window.add(left_frame, weight=1)
    
    transcript_frame = ttk.LabelFrame(left_frame, text="Real-time Transcript")
    transcript_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    transcript_text = scrolledtext.ScrolledText(
        transcript_frame, 
        wrap=tk.WORD, 
        font=("Segoe UI", 10),
        background=colors['bg_light'],
        foreground=colors['text_light'],
        insertbackground=colors['accent2'],  # Cursor color
        padx=8,
        pady=8,
        state=tk.DISABLED
    )
    transcript_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Middle Panel (Sentiment Analysis as Conversation) with modern styling
    middle_frame = ttk.Frame(paned_window)
    paned_window.add(middle_frame, weight=1)
    
    sentiment_frame = ttk.LabelFrame(middle_frame, text="Conversation with Sentiment Analysis")
    sentiment_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    sentiment_text = scrolledtext.ScrolledText(
        sentiment_frame, 
        wrap=tk.WORD, 
        font=("Segoe UI", 10),
        background=colors['bg_light'],
        foreground=colors['text_light'],
        insertbackground=colors['accent2'],  # Cursor color
        padx=8,
        pady=8,
        state=tk.DISABLED
    )
    sentiment_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Configure tags for formatting sentiment text with modern colors
    sentiment_text.tag_configure("customer", font=("Segoe UI", 10, "bold"), foreground=colors['customer'])
    sentiment_text.tag_configure("bot", font=("Segoe UI", 10, "bold"), foreground=colors['bot'])
    sentiment_text.tag_configure("positive", foreground=colors['positive'])
    sentiment_text.tag_configure("negative", foreground=colors['negative'])
    sentiment_text.tag_configure("neutral", foreground=colors['neutral'])
    
    # Right Panel (Visualization) with modern styling
    right_frame = ttk.Frame(paned_window)
    paned_window.add(right_frame, weight=1)
    
    graph_frame = ttk.LabelFrame(right_frame, text="Sentiment Visualization")
    graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Create matplotlib figure and canvas with custom styling
    fig = Figure(figsize=(5, 4), dpi=100, facecolor=colors['bg_medium'])
    ax = fig.add_subplot(111)
    
    # Style the plot for dark theme
    ax.set_facecolor(colors['bg_light'])
    fig.patch.set_facecolor(colors['bg_medium'])
    
    # Set text colors for dark theme
    ax.title.set_color(colors['text_light'])
    ax.xaxis.label.set_color(colors['text_light'])
    ax.yaxis.label.set_color(colors['text_light'])
    ax.tick_params(colors=colors['text_light'])
    for spine in ax.spines.values():
        spine.set_edgecolor(colors['text_light'])
    
    # Initial plot setup
    ax.set_title("Conversation Sentiment", fontsize=14, fontweight='bold')
    ax.set_xlabel("Message Number", fontsize=12, fontweight='bold')
    ax.set_ylabel("Sentiment Score", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color=colors['text_light'])
    ax.set_ylim(-1.1, 1.1)
    
    # Add color bands for sentiment regions with modern colors
    ax.axhspan(0.05, 1.0, alpha=0.2, color=colors['positive'])
    ax.axhspan(-0.05, 0.05, alpha=0.1, color=colors['neutral'])
    ax.axhspan(-1.0, -0.05, alpha=0.2, color=colors['negative'])
    
    # Create canvas
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Add a tight layout to the figure
    fig.tight_layout()
    
    # Footer with credits
    footer_frame = ttk.Frame(main_container)
    footer_frame.pack(fill=tk.X, pady=(15, 0))
    
    footer_label = ttk.Label(
        footer_frame, 
        text="© 2025 QA-BOT | Voice and Sentiment Analysis Tool", 
        font=("Segoe UI", 8),
        foreground=colors['accent2']
    )
    footer_label.pack(side=tk.RIGHT)
    
    version_label = ttk.Label(
        footer_frame, 
        text="v2.2 Precise Analysis Edition", 
        font=("Segoe UI", 8),
        foreground=colors['accent1']
    )
    version_label.pack(side=tk.LEFT)
    
    return root

# ========== Main Application ==========
if __name__ == "__main__":
    # Global variables
    transcriber = None
    conversation_analyzer = None
    ui_update_running = False
    
    # Create and start GUI
    root = create_gui()
    root.mainloop()