import pyrealsense2 as rs

# 启动 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# 获取深度相机的内参
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# 输出内参
print("Depth Camera Intrinsics:")
print(f"  Width: {depth_intrinsics.width}")
print(f"  Height: {depth_intrinsics.height}")
print(f"  Focal Length (fx, fy): ({depth_intrinsics.fx}, {depth_intrinsics.fy})")
print(f"  Principal Point (ppx, ppy): ({depth_intrinsics.ppx}, {depth_intrinsics.ppy})")
print(f"  Distortion Model: {depth_intrinsics.model}")
print(f"  Distortion Coefficients: {depth_intrinsics.coeffs}")

# 停止管道
pipeline.stop()
