# In this file, we define download_model
# It runs during container build time to get model weights locally
import clip

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model, transform = clip.load("ViT-B/32", device="cpu")
    print('Downloaded Model')

if __name__ == "__main__":
    download_model()
