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
        
        const response = await fetch(`${baseUrl}/ticket-management/tickets`);
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

        let today = new Date();
        let expiryDate = new Date(ticket.expiry);
        let daysUntilExpiry = Math.ceil((expiryDate - today) / (1000 * 60 * 60 * 24));

        const { className, text } = getStatusClassAndText(ticket.status, daysUntilExpiry);

        row.innerHTML = `
            <td>${ticket.vehicle_plate}</td>
            <td>${ticket.init_date}</td>
            <td>${ticket.expiry}</td>
            <td><p class="status ${className}">${text}</p></td>
            <td>${daysUntilExpiry}</td>
            <td>
                <button class="edit-btn"><i class="fa-solid fa-pen-to-square"></i></button>
                <button class="delete-btn"><i class="fa-solid fa-trash"></i></button>
            </td>
        `;
        tableBody.appendChild(row);
        row.querySelector('.edit-btn').addEventListener('click', (event) => {
            event.stopPropagation();
            viewTicketDetails(ticket.vehicle_plate);
        });

        row.querySelector('.delete-btn').addEventListener('click', (event) => {
            event.stopPropagation();
            handleTicketDelete(ticket.vehicle_plate);
        });
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
    const cccd = document.getElementById('cccd').value;

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
        cccd: cccd, // Include CCCD in the request
        status: 0
    };

    try {
        const response = await fetch(`${baseUrl}/ticket-management/ticket_admin`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();
        if (response.ok) {
            alert('Ticket added successfully');
            document.getElementById('addTicketForm').style.display = 'none';
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

        var daysUntilExpiry = Math.ceil((new Date(ticket.expiry) - new Date()) / (1000 * 60 * 60 * 24));

        document.querySelector('input[name="vehiclePlate_view"]').value = ticket.vehicle_plate;
        document.querySelector('input[name="initDate_view"]').value = ticket.init_date;
        document.querySelector('input[name="expiry_view"]').value = ticket.expiry;
        document.querySelector('input[name="status_view"]').value = getStatusClassAndText(ticket.status,daysUntilExpiry).text;
        
        const expiryDate = new Date(ticket.expiry);
        const today = new Date();

        const giaHanButton = document.getElementById('giahan');
        
        if (ticket.status === -1) {
            giaHanButton.value = "Phê duyệt";
            giaHanButton.style.display = "block";
        } else if (ticket.status === 1 || (ticket.status === 0 && today > expiryDate)) {
            giaHanButton.value = "Gia hạn";
            giaHanButton.style.display = "block";
        } else if (ticket.status === 0 && today <= expiryDate) {
            giaHanButton.style.display = "none";
        } else {
            giaHanButton.style.display = "block";
            giaHanButton.value = "Gia hạn";
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
    const vehicle_plate = formData.get('vehiclePlate_view');
    const init_date = new Date().toISOString().split('T')[0];
    const expiry = formData.get('expiry_view');
    const status = formData.get('status_view');
    let newStatus = status;
    let updateExpiry = expiry;

    if (status === 'Chưa duyệt') {
        newStatus = 0; // Chuyển sang trạng thái đã duyệt
    } else if (status === 'Hết hạn') {
        newStatus = 0; // Gia hạn
        updateExpiry = new Date();
        updateExpiry.setFullYear(updateExpiry.getFullYear() + 1); // Gia hạn thêm 1 năm (có thể thay đổi logic gia hạn)
        updateExpiry = updateExpiry.toISOString().split('T')[0];
    } else if (status === 'Chờ gia hạn') {
        newStatus = 0; // Chuyển sang trạng thái đã duyệt
    }

    const data = {
        vehicle_plate: vehicle_plate,
        init_date: init_date,
        expiry: updateExpiry,
        status: newStatus
    };

    let method = 'PUT';
    let url = `${baseUrl}/ticket-management/ticket/${vehicle_plate}`;

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

async function handleTicketDelete(vehicle_plate) {
    if (!confirm("Are you sure you want to delete this ticket?")) {
        return;
    }

    try {
        const response = await fetch(`${baseUrl}/ticket-management/ticket/${vehicle_plate}`, {
            method: 'DELETE',
        });

        if (response.ok) {
            alert('Ticket deleted successfully');
            fetchTickets();
        } else {
            const result = await response.json();
            alert(`Error: ${result.message}`);
        }
    } catch (error) {
        console.error('Error deleting ticket:', error);
    }
}