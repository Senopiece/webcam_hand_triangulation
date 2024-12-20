import argparse
import cv2
import asyncio
from async_hands import (
    HandTrackersPool,
    LiveStreamAsyncHands,
    AsyncHandsThreadedVideo,
    AsyncHandsThreadedImage,
    AsyncHandsThreadedBuildinSolution,
)
from fps_counter import FPSCounter


impl_selector = {
    "LiveStreamAsyncHands": lambda: LiveStreamAsyncHands(),
    "LiveStreamAsyncHands(GPU)": lambda: LiveStreamAsyncHands(gpu=True),
    "AsyncHandsThreadedVideo": lambda: AsyncHandsThreadedVideo(),
    "AsyncHandsThreadedVideo(GPU)": lambda: AsyncHandsThreadedVideo(gpu=True),
    "AsyncHandsThreadedImage": lambda: AsyncHandsThreadedImage(),
    "AsyncHandsThreadedImage(GPU)": lambda: AsyncHandsThreadedImage(gpu=True),
    "AsyncHandsThreadedBuildinSolution": lambda: AsyncHandsThreadedBuildinSolution(),
}


async def process_video(
    channels=4,
    division=2,
    impl: str = "AsyncHandsThreadedBuildinSolution",
    video_path: str = "test.mkv",
):
    # Initialize
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    processors = [
        HandTrackersPool([impl_selector[impl]() for _ in range(division)])
        for _ in range(channels)
    ]

    fps = FPSCounter()

    run = True

    async def consuming_loop():
        while (
            run
            or any(
                not processor.idle_workers.full() for processor in processors
            )  # any channel is in processing -> new results may arrive
            or any(
                not processor.results.empty() for processor in processors
            )  # any channel has non empty results -> need to process them
        ):
            # NOTE: it will hang freeing if channels got not equal amounts of .send calls
            await asyncio.gather(*[processor.results.get() for processor in processors])
            fps.count()

    async def feeding_loop():
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Error reading frames.")
                break

            await asyncio.gather(*[processor.send(frame) for processor in processors])

    # Run loops: consume asyncronusly and join with feeding
    consuming_task = asyncio.create_task(consuming_loop())
    await feeding_loop()

    # Finalize
    run = False  # notify consuming to stop
    await consuming_task  # wait for it to finish

    # Print metrics
    fps.mean()

    # Release resources
    await asyncio.gather(*[processor.dispose() for processor in processors])
    cap.release()


async def main():
    parser = argparse.ArgumentParser(description="Process a video using async hands.")
    parser.add_argument(
        "--video_path", type=str, default="test.mkv", help="Path to the video file."
    )
    parser.add_argument(
        "--channels", type=int, default=4, help="Number of channels to process."
    )
    parser.add_argument(
        "--division", type=int, default=2, help="Size of pool of workers per channel."
    )
    parser.add_argument(
        "--impl",
        type=str,
        default="AsyncHandsThreadedBuildinSolution",
        choices=list(impl_selector.keys()),
        help="Implementation to use for processing.",
    )

    args = parser.parse_args()

    try:
        await process_video(
            channels=args.channels,
            division=args.division,
            impl=args.impl,
            video_path=args.video_path,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    asyncio.run(main())
