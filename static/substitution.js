// substitution.js

// Listen for input changes in the ingredient input field
document.getElementById('ingredient-input').addEventListener('input', function() {
    var ingredient = document.getElementById('ingredient-input').value.trim();

    // Send an AJAX request to the server to get substitutions for the entered ingredient
    $.ajax({
        url: '/get_substitutions',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ ingredient: ingredient }),
        success: function(response) {
            // Update the dropdown menu with substitutions
            updateSubstitutionDropdown(response);
        },
        error: function(xhr, status, error) {
            console.error('Error:', error);
        }
    });
});

// Function to update the dropdown menu with substitutions
function updateSubstitutionDropdown(substitutions) {
    var dropdown = document.getElementById('substitution-dropdown');
    dropdown.innerHTML = ''; // Clear existing options

    // Add new options for each substitution
    substitutions.forEach(function(substitution) {
        var option = document.createElement('option');
        option.value = substitution;
        option.text = substitution;
        dropdown.appendChild(option);
    });
}
