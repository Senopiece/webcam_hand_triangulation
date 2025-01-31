from typing import Any, Dict
from finalizable_queue import EmptyFinalized, FinalizableQueue


def ordering_loop(
        in_queue: FinalizableQueue,
        out_queue: FinalizableQueue,
    ):
    expecting = 0
    unordered: Dict[int, Any] = {}
    while True:
        try:
            elem = in_queue.get()
        except EmptyFinalized:
            break

        index: int = elem[0]
        data: Any = elem[1]

        if expecting == index:
            out_queue.put(data)
            while True:
                expecting += 1
                data = unordered.get(expecting, None)
                if data is None:
                    break
                else:
                    del unordered[expecting]
                    out_queue.put(data)
        else:
            unordered[index] = data
        
        in_queue.task_done()
    
    out_queue.finalize()
    print("A ordering loop finished.")