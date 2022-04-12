echo "===== Sorting imports ====="

isort --trailing-comma --line-width 120 --multi-line 3 gym_db/
isort --trailing-comma --line-width 120 --multi-line 3 swirl/
isort --trailing-comma --line-width 120 --multi-line 3 *.py

echo ""
echo "===== Formatting via black ====="

black --line-length 120 gym_db/
black --line-length 120 swirl/
black --line-length 120 *.py
