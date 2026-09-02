from typing import Iterable, Optional, Any
import sys
from tqdm import tqdm
import comfy.utils


class cqdm:
    """Combined progress bar wrapper unifying tqdm console output and ComfyUI GUI ProgressBar."""

    def __init__(
        self,
        iterable: Optional[Iterable[Any]] = None,
        total: Optional[int] = None,
        desc: str = "Processing",
        disable: bool = False,
        **kwargs: Any,
    ) -> None:
        self.iterable = iterable
        self.total = total
        self.desc = desc

        if iterable is not None and total is None:
            try:
                self.total = len(iterable)  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                self.total = None

        self.pbar: Optional[comfy.utils.ProgressBar] = (
            comfy.utils.ProgressBar(self.total) if self.total is not None else None
        )

        self.tqdm: Optional[tqdm] = tqdm(
            iterable=self.iterable,
            total=self.total,
            desc=self.desc,
            disable=disable,
            dynamic_ncols=True,
            file=sys.stdout,
            **kwargs,
        )

    def __iter__(self):
        if self.tqdm is None:
            return
        for item in self.tqdm:
            if self.pbar:
                self.pbar.update(1)
            yield item

    def update(self, n: int = 1) -> None:
        if self.tqdm:
            self.tqdm.update(n)
        if self.pbar:
            self.pbar.update(n)

    def set_description(self, desc: str) -> None:
        if self.tqdm:
            self.tqdm.set_description(desc)

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        if self.tqdm:
            self.tqdm.set_postfix(*args, **kwargs)

    def close(self) -> None:
        if self.tqdm is not None:
            try:
                self.tqdm.close()
            except Exception as e:
                print(f"[llama-cpp_vlm] Warning closing tqdm: {e}")
            self.tqdm = None

    def __enter__(self) -> "cqdm":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __len__(self) -> int:
        return self.total if self.total is not None else 0