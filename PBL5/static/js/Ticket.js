const baseUrl = 'http://192.168.174.130:5000';
async function fetchTickets() {
    try {
        const response = await fetch(`${baseUrl}/ticket-management/tickets`);
        const data = await response.json();
        populateTicketTable(data);
    } catch (error) {
        console.error('Error fetching ticket data:', error);
    }
}

function populateTicketTable(tickets) {
    const tableBody = document.getElementById('ticketTableBody');
    tableBody.innerHTML = '';

    tickets.forEach(ticket => {
        const row = document.createElement('tr');
        row.setAttribute('data-ticket-id', ticket.vehicle_plate);
        row.innerHTML = `
            <td>${ticket.vehicle_plate}</td>
            <td>${ticket.init_date}</td>
            <td>${ticket.expiry}</td>
            <td>${ticket.id_customer}</td>
        `;
        row.addEventListener('click', () => viewTicketDetails(ticket.vehicle_plate));
        tableBody.appendChild(row);
    });
}

async function createTicket(event) {
    event.preventDefault();
    const vehiclePlate = document.getElementById('vehiclePlate').value;
    const initDate = new Date().toISOString().split('T')[0];
    const expiry = document.getElementById('expiry').value;
    const customerId = document.getElementById('customerId').value;

    const data = {
        vehicle_plate: vehiclePlate,
        init_date: initDate,
        expiry: expiry,
        id_customer: customerId
    };

    try {
        const response = await fetch(`${baseUrl}/ticket-management/ticket`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();
        if (response.ok) {
            alert('Ticket added successfully');
            document.getElementById('addTicketForm').classList.add('hidden');
            document.querySelector('.header').classList.remove('hidden');
            document.querySelector('.table__body').classList.remove('hidden');
            fetchTickets();
        } else {
            alert(`Error: ${result.message}`);
        }
    } catch (error) {
        console.error('Error adding ticket:', error);
    }
}

async function viewTicketDetails(ticketId) {
    try {
        const response = await fetch(`${baseUrl}/ticket-management/ticket/${ticketId}`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const ticket = await response.json();
        document.querySelector('input[name="vehiclePlate_view"]').value = ticket.vehicle_plate;
        document.querySelector('input[name="initDate_view"]').value = ticket.init_date;
        document.querySelector('input[name="expiry_view"]').value = ticket.expiry;
        document.querySelector('input[name="customerId_view"]').value = ticket.id_customer;

        document.getElementById('detailTicketForm').classList.remove('hidden');
        document.querySelector('.header').classList.add('hidden');
        document.querySelector('.table__body').classList.add('hidden');
    } catch (error) {
        console.error('Error fetching ticket details:', error);
    }
}

function toggleAddTicketForm() {
    document.getElementById('addTicketForm').classList.remove('hidden');
    document.getElementById('detailTicketForm').classList.add('hidden');
    document.querySelector('.header').classList.add('hidden');
    document.querySelector('.table__body').classList.add('hidden');

}

async function handleTicketAction(event, formElement, action) {
    event.preventDefault();
    const formData = new FormData(formElement);
    const data = {
        vehicle_plate: formData.get('vehiclePlate_view'),
        init_date: formData.get('initDate_view'),
        expiry: formData.get('expiry_view'),
        id_customer: formData.get('customerId_view'),
    };
    let method = 'PUT';
    let url = `${baseUrl}/ticket-management/ticket/${data.vehicle_plate}`;

    if (action === "Remove") {
        method = 'DELETE';
    }
    try {
        const response = await fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            },
            body: action === "Remove" ? null : JSON.stringify(data),
        });
        const result = await response.json();
        if (response.ok) {
            if (action === "Remove") {
                alert(`Delete successful`);
            } else {
                alert(`Update successful`);
            }
            document.getElementById('detailTicketForm').classList.add('hidden');
            document.querySelector('.header').classList.remove('hidden');
            document.querySelector('.table__body').classList.remove('hidden');
            fetchTickets();
        } else {
            alert(`Error: ${result.message}`);
        }
    } catch (error) {
        console.error(`Error ${action} ticket:`, error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    fetchTickets();
    const searchInput = document.querySelector('.input-group input');
    searchInput.addEventListener('input', searchTable);
});

function searchTable() {
    let tableRows = document.querySelectorAll('tbody tr');
    tableRows.forEach((row, i) => {
        let tableData = row.textContent.toLowerCase();
        let searchData = searchInput.value.toLowerCase();
        row.classList.toggle('hide', tableData.indexOf(searchData) < 0);
        row.style.setProperty('--delay', i / 25 + 's');
    });
}