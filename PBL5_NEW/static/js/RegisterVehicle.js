// const baseUrl = 'http://127.0.0.1:5000';
document.addEventListener('DOMContentLoaded', () => {
    fetchTickets();
});

var search = document.querySelector('.input-group input'),
    table_rows = document.querySelectorAll('tbody tr'),
    table_headings = document.querySelectorAll('thead th');

async function fetchTickets() {
    try {
        const customerId = await getSessionData();
        if (!customerId) {
            alert('Could not retrieve customer ID from session.');
            return;
        }
        
        const response = await fetch(`${baseUrl}/ticket-management/tickets/customer/${customerId}`);
        const data = await response.json();
        populateTicketTable(data);
    } catch (error) {
        console.error('Error fetching ticket data:', error);
    }
}

function getStatusClassAndText(status, daysUntilExpiry) {
    if (status === -1) {
        return { className: 'cancelled', text: 'Chưa duyệt' };
    } else if (status === 0 && daysUntilExpiry <= 0) {
        return { className: 'shipped', text: 'Hết hạn' };
    } else if (status === 0) {
        return { className: 'delivered', text: 'Đã duyệt' };
    } else if (status === 1) {
        return { className: 'pending', text: 'Chờ gia hạn' };
    }
    return { className: '', text: '' };
}

function populateTicketTable(tickets) {
    const tableBody = document.getElementById('ticketTableBody');
    tableBody.innerHTML = '';

    tickets.forEach(ticket => {
        const row = document.createElement('tr');
        row.setAttribute('data-ticket-id', ticket.vehicle_plate);

        const today = new Date();
        const expiryDate = new Date(ticket.expiry);
        const daysUntilExpiry = Math.ceil((expiryDate - today) / (1000 * 60 * 60 * 24));

        const { className, text } = getStatusClassAndText(ticket.status, daysUntilExpiry);

        row.innerHTML = `
            <td>${ticket.vehicle_plate}</td>
            <td>${ticket.init_date}</td>
            <td>${ticket.expiry}</td>
            <td><p class="status ${className}">${text}</p></td>
            <td>${daysUntilExpiry}</td>
        `;
        row.addEventListener('click', () => viewTicketDetails(ticket.vehicle_plate));
        tableBody.appendChild(row);
    });
    table_headings = document.querySelectorAll('thead th');
    table_rows = document.querySelectorAll('tbody tr');
    table_headings.forEach((head, i) => {
        let sort_asc = true;
        head.onclick = () => {
            table_headings.forEach(head => head.classList.remove('active'));
            head.classList.add('active');
    
            document.querySelectorAll('td').forEach(td => td.classList.remove('active'));
            table_rows.forEach(row => {
                row.querySelectorAll('td')[i].classList.add('active');
            })
    
            head.classList.toggle('asc', sort_asc);
            sort_asc = head.classList.contains('asc') ? false : true;
    
            sortTable(i, sort_asc);
        }
    })
}


async function getSessionData() {
    try {
        const response = await fetch('/get_session');
        if (response.ok) {
            const sessionData = await response.json();
            return sessionData.id_account;
        } else {
            console.error('No session data found');
            return null;
        }
    } catch (error) {
        console.error('Error fetching session data:', error);
        return null;
    }
}

async function createTicket(event) {
    event.preventDefault();
    const vehiclePlate = document.getElementById('vehiclePlate').value;
    const initDate = new Date().toISOString().split('T')[0];
    const expiry = document.getElementById('expiry').value;
    const customerId = await getSessionData();
    if (!customerId) {
        alert('Could not retrieve customer ID from session.');
        return;
    }
    // Convert the dates to Date objects
    const initDateObj = new Date(initDate);
    const expiryDateObj = new Date(expiry);
    
    // Check if expiry is before initDate
    if (expiryDateObj < initDateObj) {
        document.getElementById('error-message').textContent = 'Expiry date cannot be before the initial date.';
        return;
    } 

    const data = {
        vehicle_plate: vehiclePlate,
        init_date: initDate,
        expiry: expiry,
        id_customer: customerId,
        status : -1
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
            document.getElementById('customers_table').style.display = 'block';
            fetchTickets();
        } else {
            document.getElementById('error-message').textContent = result.message;
        }
    } catch (error) {
        console.error('Error adding ticket:', error);
    }
}







async function viewTicketDetails(vehicle_plate) {
    try {
        const response = await fetch(`${baseUrl}/ticket-management/ticket/${vehicle_plate}`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const ticket = await response.json();
        document.querySelector('input[name="vehiclePlate_view"]').value = ticket.vehicle_plate;
        document.querySelector('input[name="initDate_view"]').value = ticket.init_date;
        document.querySelector('input[name="expiry_view"]').value = ticket.expiry;
        
        const expiryDate = new Date(ticket.expiry);
        const today = new Date();

        const giaHanButton = document.getElementById('giahan');
        
        if (today > expiryDate) {
            giaHanButton.disabled = false;
        } else {
            giaHanButton.disabled = true;
            giaHanButton.value = "Chưa hết hạn"
        }

        document.getElementById('detailTicketForm').classList.remove('hidden');
        document.getElementById('customers_table').style.display = 'none';
    } catch (error) {
        console.error('Error fetching ticket details:', error);
    }
}

function toggleAddTicketForm() {
    document.getElementById('addTicketForm').classList.remove('hidden');
    document.getElementById('detailTicketForm').classList.add('hidden');
    document.getElementById('customers_table').style.display = 'none';
}



async function handleTicketAction(event, formElement, action) {
    event.preventDefault();
    const formData = new FormData(formElement);
    const data = {
        vehicle_plate: formData.get('vehiclePlate_view'),
        init_date: new Date().toISOString().split('T')[0],
        expiry: formData.get('expiry_view'),
        status : 1,
    };
    let method = 'PUT';
    let url = `${baseUrl}/ticket-management/ticket/${data.vehicle_plate}`;

    if (action === "Xóa") {
        method = 'DELETE';
    }
    try {
        const response = await fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            },
            body: action === "Xóa" ? null : JSON.stringify(data),
        });
        const result = await response.json();
        if (response.ok) {
            if (action === "Xóa") {
                alert(`Delete successful`);
            } else {
                alert(`Update successful`);
            }
            document.getElementById('detailTicketForm').classList.add('hidden');
            document.getElementById('customers_table').style.display = 'block';
            fetchTickets();
        } else {
            alert(`Error: ${result.message}`);
        }
    } catch (error) {
        console.error(`Error ${action} ticket:`, error);
    }
}



// 1. Searching for specific data of HTML table
search.addEventListener('input', searchTable);

function searchTable() {
    var table_rows = document.querySelectorAll('tbody tr');
    table_rows.forEach((row, i) => {
        let table_data = row.textContent.toLowerCase(),
            search_data = search.value.toLowerCase();

        row.classList.toggle('hide', table_data.indexOf(search_data) < 0);
        row.style.setProperty('--delay', i / 25 + 's');
    })

    document.querySelectorAll('tbody tr:not(.hide)').forEach((visible_row, i) => {
        visible_row.style.backgroundColor = (i % 2 == 0) ? 'transparent' : '#0000000b';
    });
}

function sortTable(column, sort_asc) {
    [...table_rows].sort((a, b) => {
        let first_row = a.querySelectorAll('td')[column].textContent.toLowerCase(),
            second_row = b.querySelectorAll('td')[column].textContent.toLowerCase();

        return sort_asc ? (first_row < second_row ? 1 : -1) : (first_row < second_row ? -1 : 1);
    })
        .map(sorted_row => document.querySelector('tbody').appendChild(sorted_row));
}

