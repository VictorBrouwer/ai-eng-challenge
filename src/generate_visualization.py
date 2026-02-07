import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from src.graph.builder import build_graph
    
    print("Building graph...")
    graph = build_graph()
    
    print("Generating Mermaid PNG...")
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open("graph_visualization.png", "wb") as f:
            f.write(png_data)
        print("Success: Saved to graph_visualization.png")
    except Exception as e:
        print(f"Failed to generate PNG: {e}")
        print("Generating Mermaid text instead...")
        print(graph.get_graph().draw_mermaid())
        
except Exception as e:
    print(f"Error: {e}")
