import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def create_kafka_nifi_diagram(save_path=None):
    """
    Create a Kafka-NiFi data pipeline diagram
    
    Args:
        save_path (str): Path to save the diagram. If None, displays only.
    """
    # Create figure with higher DPI for better quality
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    
    # Enhanced color scheme
    colors = {
        "sources": "#E3F2FD",      # Light blue
        "kafka": "#FFF3E0",        # Light orange
        "nifi": "#E8F5E8",         # Light green
        "targets": "#F3E5F5"       # Light purple
    }
    
    # Main components with improved positioning
    components = {
        "Sources\n(Files, DBs, APIs)": (2, 3.5, "sources"),
        "Apache Kafka\n(Message Broker)": (5, 3.5, "kafka"),
        "Apache NiFi\n(Data Processing)": (8, 3.5, "nifi"),
        "Targets\n(Data Lakes, DBs)": (10.5, 3.5, "targets")
    }
    
    # Draw components with enhanced styling
    for label, (x, y, color_key) in components.items():
        # Main box
        box = mpatches.FancyBboxPatch(
            (x-1.1, y-0.8), 2.2, 1.6,
            boxstyle="round,pad=0.15",
            edgecolor="#2C3E50", 
            facecolor=colors[color_key],
            linewidth=2
        )
        ax.add_patch(box)
        
        # Text
        ax.text(x, y, label, ha="center", va="center", 
                fontsize=11, weight="bold", color="#2C3E50")
    
    # Enhanced arrows with labels
    arrow_style = dict(arrowstyle="->", lw=3, color="#34495E")
    
    # Source to Kafka
    ax.annotate("", xy=(3.8, 3.5), xytext=(3.2, 3.5), arrowprops=arrow_style)
    ax.text(3.5, 4.1, "Ingest", ha="center", fontsize=9, color="#7F8C8D", style="italic")
    
    # Kafka to NiFi
    ax.annotate("", xy=(6.8, 3.5), xytext=(6.2, 3.5), arrowprops=arrow_style)
    ax.text(6.5, 4.1, "Stream", ha="center", fontsize=9, color="#7F8C8D", style="italic")
    
    # NiFi to Targets
    ax.annotate("", xy=(9.3, 3.5), xytext=(9.2, 3.5), arrowprops=arrow_style)
    ax.text(9.8, 4.1, "Process & Load", ha="center", fontsize=9, color="#7F8C8D", style="italic")
    
    # Enhanced title with subtitle
    ax.text(6, 6.2, "Real-Time Data Pipeline Architecture", 
            ha="center", va="center", fontsize=16, weight="bold", color="#2C3E50")
    ax.text(6, 5.7, "Kafka + NiFi Integration for Streaming Data Processing", 
            ha="center", va="center", fontsize=12, color="#7F8C8D", style="italic")
    
    # Add flow indicators
    flow_components = [
        ("Real-time\nData Streams", 2, 1.8),
        ("Message\nQueuing", 5, 1.8),
        ("ETL\nProcessing", 8, 1.8),
        ("Persistent\nStorage", 10.5, 1.8)
    ]
    
    for desc, x, y in flow_components:
        ax.text(x, y, desc, ha="center", va="center", 
                fontsize=9, color="#95A5A6", style="italic")
    
    # Add border
    border = mpatches.Rectangle((0.5, 0.5), 11, 6, fill=False, 
                               edgecolor="#BDC3C7", linewidth=1)
    ax.add_patch(border)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Diagram saved to: {save_path}")
    
    plt.show()
    return fig

# Create and save the diagram
if __name__ == "__main__":
    # Option 1: Save to current directory
    create_kafka_nifi_diagram("kafka_nifi_pipeline.png")
    
    # Option 2: Save to specific path (uncomment and modify as needed)
    # create_kafka_nifi_diagram("/path/to/your/downloads/kafka_nifi_pipeline.png")
    
    # Option 3: Save multiple formats
    formats = ['png', 'pdf', 'svg']
    for fmt in formats:
        create_kafka_nifi_diagram(f"kafka_nifi_pipeline.{fmt}")
        
    print("All diagrams created successfully!")
    print("\nFiles created:")
    for fmt in ['png', 'pdf', 'svg']:
        if os.path.exists(f"kafka_nifi_pipeline.{fmt}"):
            print(f"  - kafka_nifi_pipeline.{fmt}")
