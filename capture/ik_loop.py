import torch
import numpy as np
from .kinematics import inverse_hand_kinematics
from .finalizable_queue import EmptyFinalized, FinalizableQueue


def ik_loop(
    in_queue: FinalizableQueue,
    out_queue: FinalizableQueue,
):
    while True:
        try:
            indexes = []
            fpses = []
            debts = []
            hand_points_batch = []
            batch = in_queue.get_all_waiting()
            for elem in batch:
                index = elem[0]
                hand_points, coupling_fps, debt_size = elem[1]
                indexes.append(index)
                hand_points_batch.append(hand_points)
                fpses.append(coupling_fps)
                debts.append(debt_size)
                in_queue.task_done()
        except EmptyFinalized:
            break

        non_none_hand_points_batch = [e for e in hand_points_batch if e is not None]
        full_angles_batch = [None] * len(hand_points_batch)
        if len(non_none_hand_points_batch) != 0:
            non_none_hand_points_batch = torch.tensor(
                np.stack(non_none_hand_points_batch), dtype=torch.float32
            )
            angles_batch = inverse_hand_kinematics(non_none_hand_points_batch)
            angles_batch = angles_batch.numpy(force=True)

            hand_points_batch_non_null_indices = [
                i
                for i in range(len(hand_points_batch))
                if hand_points_batch[i] is not None
            ]
            for i, j in enumerate(hand_points_batch_non_null_indices):
                full_angles_batch[j] = angles_batch[i]

        for index, angles, coupling_fps, debt_size in zip(
            indexes, full_angles_batch, fpses, debts
        ):
            out_queue.put(
                (
                    index,
                    (
                        angles,
                        coupling_fps,
                        debt_size,
                    ),
                )
            )
