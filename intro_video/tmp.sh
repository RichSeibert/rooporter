for img in *.jpg; do
    ffmpeg -i "$img" -vf "scale='if(lt(a,9/16),544,-1)':'if(lt(a,9/16),-1,960)',crop=544:960" "output_${img}"
done

