import io

from langgraph.graph.state import CompiledStateGraph
from PIL import Image


def show_graph(graph: CompiledStateGraph):
    image = Image.open(io.BytesIO(graph.get_graph(xray=True).draw_mermaid_png()))
    image.show()
