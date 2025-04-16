#! /bin/bash
rsync --progress -azhe 'ssh -i ~/.ssh/mobisys2025.pem' /home/data/agile3d/carl/experiments/ agile3d@172.30.53.226:~/fastdata/agile3d/agile3d/carl/experiments/