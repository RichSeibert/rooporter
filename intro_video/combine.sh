ffmpeg -framerate 7.6923 -pattern_type glob -i "output_*.jpg" -i intro_sound.mp3 -r 24 -c:v libopenh264 -preset fast -crf 23 -b:v 758k -c:a aac -b:a 128k -shortest final_video_with_audio.mp4
