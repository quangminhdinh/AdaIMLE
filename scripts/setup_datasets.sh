 #!/bin/bash

 gdown https://drive.google.com/file/d/1VwFFzU8wJD1XJtfg60iLwnyBQ_cLZObL/view\?usp\=sharing --fuzzy
 tar -xvf datasets.tar

 wandb sync --include-offline ./wandb/offline-run-20250620_*