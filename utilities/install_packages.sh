# Iterate through each line in requirements.txt
while IFS= read -r package || [[ -n "$package" ]]; do
    # Install the package
    pip install "$package"
done < ../requirements.txt