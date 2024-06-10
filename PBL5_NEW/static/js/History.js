function fillter() {
    var container = document.querySelector(".container");
    container.style.display = "block";
}
function cancel() {
    var container = document.querySelector(".container");
    container.style.display = "none";
}


// const baseUrl = 'http://127.0.0.1:5000';

var search = document.querySelector('.input-group input'),
    table_rows = document.querySelectorAll('tbody tr'),
    table_headings = document.querySelectorAll('thead th');

document.addEventListener('DOMContentLoaded', fetchHistories);

async function fetchHistories() {
    try {
        const customerId = await getSessionData();
        if (!customerId) {
            alert('Could not retrieve customer ID from session.');
            return;
        }
        
        const response = await fetch(`${baseUrl}/history-management/histories`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        populateHistoryTable(data);
    } catch (error) {
        console.error('Error fetching history data:', error);
    }
}

function populateHistoryTable(histories) {
    const tableBody = document.getElementById('historyTableBody');
    tableBody.innerHTML = '';
    const reversedHistories = histories.slice().reverse();
    reversedHistories.forEach(history => {
        const row = document.createElement('tr');
        row.setAttribute('data-history-id', history.id_history);

        row.innerHTML = `
            <td>${history.vehicle_plate}</td>
            <td><p class="status delivered">${history.date_in || ''}</p></td>
            <td><p class="status delivered">${history.time_in || ''}</p></td>
            <td><p class="status pending">${history.date_out || ''}</p></td>
            <td><p class="status pending">${history.time_out || ''}</p></td>
        `;
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
        const response = await fetch(`${baseUrl}/get_session`);
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

document.getElementById('confirm_btn').addEventListener('click', async () => {
    const dateIn = document.querySelector('input[name="date_in"]').value;
    const dateOut = document.querySelector('input[name="date_out"]').value;
    const timeIn = document.querySelector('input[name="time_in"]').value;
    const timeOut = document.querySelector('input[name="time_out"]').value;

    try {
        const params = new URLSearchParams();
        if (dateIn) params.append('date_in', dateIn);
        if (dateOut) params.append('date_out', dateOut);
        if (timeIn) params.append('time_in', timeIn);
        if (timeOut) params.append('time_out', timeOut);

        const response = await fetch(`${baseUrl}/history-management/histories/all?${params.toString()}`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        populateHistoryTable(data);
    } catch (error) {
        console.error('Error fetching filtered history data:', error);
    }
    cancel();
});