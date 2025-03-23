dd if=/dev/zero of=testfile bs=4M count=256 oflag=direct status=progress

rm testfile