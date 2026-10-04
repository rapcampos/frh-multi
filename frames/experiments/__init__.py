# Lazy re-export: figstep_cdg_comparison uses itertools.batched (Python 3.12+),
# which would otherwise make the whole package un-importable on 3.11.
def __getattr__(name):
    if name == "MultilingualModelGenerator":
        from .figstep_cdg_comparison import MultilingualModelGenerator

        return MultilingualModelGenerator
    raise AttributeError(name)
