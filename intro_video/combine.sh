ffmpeg -framerate 7.6923 -pattern_type glob -i "output_*.jpg" -i intro_sound.mp3 -r 24 -c:v mpeg4 -q:v 2 -c:a aac -shortest final_video_with_audio.mp4
