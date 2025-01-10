# Imports as necessary
from IPython.display import display, HTML

class BtResult:
    def __init__(self) -> None:
        # Class can be static or you can hold results in an attribute
        # static is recommended to avoid instantiation. You can have additional functionality that becomes
        # available for declared instances but keeping core functionality static would be better
        ...

    @staticmethod
    def _beatuify_results(results: dict[str, float | dict]) -> HTML:
        # Better to have this static to reduce overhead by instantiating a class
        # for all iterations of a potential for loop

        template: str = ...
        ...
        return HTML(...)

    @staticmethod
    def pass_results(results: dict[...], show: bool = True) -> dict[...] | None:
        # Again, static is recommended to avoid instantiation

        if show:
            html = self._beatuify_results(results)
            display(html)
            return None

        # Format results of return as is
        else:
            ...
            return results

    # Declare properties here as necessary, avoid setters (or define setters that don't set)
    # as modifying results accidentally can be dangerous
    ...
