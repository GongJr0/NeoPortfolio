# Imports as necessary


from IPython.display import display, HTML
import matplotlib.pyplot as plt
from io import BytesIO
import base64


class BtResult:
    def __init__(self) -> None:
        # Class can be static, or you can hold results in an attribute
        # static is recommended to avoid instantiation. You can have additional functionality that becomes
        # available for declared instances but keeping core functionality static would be better
        ...

    @staticmethod
    def plot_signals(buy: dict[...], sell: dict[...]) -> BytesIO:
        # Plotting Logic
        ...

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        return buffer

    @staticmethod
    def _beatuify_results(results: dict[str, float | dict], plot: BytesIO) -> HTML:
        # Better to have this static to reduce overhead by instantiating a class
        # for all iterations of a potential for loop
        plot = base64.b64decode(plot.getvalue()).decode('utf-8')
        plot_html = f'<img src="data:image/png;base64,{plot}">'

        template: str = ...
        html = template + plot_html + ...

        return HTML(html)

    @staticmethod
    def pass_results(results: dict[...], show: bool = True) -> dict[...] | None:
        # Again, static is recommended to avoid instantiation

        if show:
            html = BtResult._beatuify_results(results)
            display(html)
            return None

        # Format results or return as is
        else:
            ...
            return results

    # Declare properties here as necessary, avoid setters (or define setters that don't set)
    # as modifying results accidentally can be dangerous
    ...
