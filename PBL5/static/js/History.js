const baseUrl = 'http://192.168.174.130:5000';
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

    const reversedHistories = histories.slice().reverse();

    reversedHistories.forEach(history => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${history.id_history}</td>
            <td>${history.vehicle_plate}</td>
            <td>${history.date_in}</td>
            <td>${history.time_in}</td>
            <td>${history.date_out}</td>
            <td>${history.time_out}</td>
        `;
        tableBody.appendChild(row);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    fetchHistories();
});