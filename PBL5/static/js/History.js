const baseUrl = 'http://10.10.58.77:5000';
async function fetchHistories() {
    try {
        const response = await fetch(`${baseUrl}/history-management/histories`);
        const data = await response.json();
        populateCustomerTable(data);
    } catch (error) {
        console.error('Error fetching customer data:', error);
    }
}

function populateCustomerTable(histories) {
    const tableBody = document.getElementById('historyTableBody');
    tableBody.innerHTML = ''; // Clear existing rows

    histories.forEach(histories => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${histories.id_history}</td>
            <td>${histories.vehicle_plate}</td>
            <td>${histories.date_in}</td>
            <td>${histories.time_in}</td>
            <td>${histories.date_out}</td>
            <td>${histories.time_out}</td>
        `;
        tableBody.appendChild(row);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    fetchHistories();
});