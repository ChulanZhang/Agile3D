#! /bin/bash
rsync --progress -azhe 'ssh -i ~/.ssh/mobisys2025.pem' /home/data/agile3d/output/exp1/ agile3d@172.30.166.233:/home/pengcheng/data/agile3d/output/exp1/
rsync --progress -azhe 'ssh -i ~/.ssh/mobisys2025.pem' /home/data/agile3d/output/exp3/ agile3d@172.30.166.233:/home/pengcheng/data/agile3d/output/exp3/
rsync --progress -azhe 'ssh -i ~/.ssh/mobisys2025.pem' agile3d@172.30.166.233:/home/pengcheng/data/agile3d/output/exp1/ /home/data/agile3d/output/exp1/
rsync --progress -azhe 'ssh -i ~/.ssh/mobisys2025.pem' agile3d@172.30.166.233:/home/pengcheng/data/agile3d/output/exp3/ /home/data/agile3d/output/exp3/