import config
import utils

def main():
    utils.set_seed(config.RANDOM_SEED)
    utils.print_header()
    
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Models directory: {config.MODELS_DIR}")
    print("\n Your environment is ready!")

if __name__ == "__main__":
    main()
