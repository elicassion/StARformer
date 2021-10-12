for game in 'Breakout' 'Qbert' 'Seaquest'
do
    echo "game $game"
    python run_star_atari.py --seed 123 --epochs 10 --model_type 'star_rwd' --num_steps 500000 --num_buffers 50 --game $game --batch_size 64
done