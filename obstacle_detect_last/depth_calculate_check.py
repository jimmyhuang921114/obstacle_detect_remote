import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

# 选择适合的分辨率 (RGB和深度可不同)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline.start(config)
profile = pipeline.get_active_profile()

# 获取深度相机内参
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
depth_intrinsics = depth_stream.get_intrinsics()

# 获取RGB相机内参
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
color_intrinsics = color_stream.get_intrinsics()

print(f"🔹 深度相机内参: fx={depth_intrinsics.fx}, fy={depth_intrinsics.fy}, "
      f"cx={depth_intrinsics.ppx}, cy={depth_intrinsics.ppy}")

print(f"🔸 RGB相机内参: fx={color_intrinsics.fx}, fy={color_intrinsics.fy}, "
      f"cx={color_intrinsics.ppx}, cy={color_intrinsics.ppy}")

pipeline.stop()
