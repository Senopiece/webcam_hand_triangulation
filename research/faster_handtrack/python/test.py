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
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    fps = FPSCounter()

    processors = [
        HandTrackersPool([impl_selector[impl]() for _ in range(division)])
        for _ in range(channels)
    ]

    async def consuming_loop():
        while True:
            await asyncio.gather(*[processor.results.get() for processor in processors])
            fps.count()

    async def feeding_loop():
        while cap.isOpened():
            ret, frame = cap.read()
            ts = cap.get(cv2.CAP_PROP_POS_MSEC)

            if not ret:
                print("Error reading frames.")
                break

            await asyncio.gather(
                *[processor.send(ts, frame) for processor in processors]
            )

    consuming_task = asyncio.create_task(consuming_loop())
    await feeding_loop()

    consuming_task.cancel()

    fps.mean()

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
