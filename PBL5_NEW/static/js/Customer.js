// const baseUrl = 'http://127.0.0.1:5000';
async function fetchCustomers() {
    try {
        const response = await fetch(`${baseUrl}/information-management/informations`);
        const data = await response.json();
        populateCustomerTable(data);
    } catch (error) {
        console.error('Error fetching customer data:', error);
    }
}

function populateCustomerTable(customers) {
    const tableBody = document.getElementById('customerTableBody');
    tableBody.innerHTML = ''; // Clear existing rows

    customers.forEach(customer => {
        const row = document.createElement('tr');
        row.setAttribute('data-customer-id', customer.id_account);
        row.innerHTML = `
            <td>${customer.name}</td>
            <td>${customer.gender ? 'Male' : 'Female'}</td>
            <td>${customer.phone}</td>
            <td>${customer.gmail}</td>
            <td>${customer.cccd}</td>
        `;
        row.addEventListener('click', () => viewCustomerDetails(customer.id_account));
        tableBody.appendChild(row);
    });
}

async function createCustomer(event) {
    event.preventDefault();
    const name = document.getElementById('name').value;
    const phone = document.getElementById('phone').value;
    const cccd = document.getElementById('cccd').value;
    const gender = document.querySelector('input[name="gender"]:checked').value;

    try {
        const response = await fetch(`${baseUrl}/customer-management/customer`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name, phone, cccd, gender: gender === 'Male' }),
        });

        const data = await response.json();
        if (response.ok) {
            alert('Customer added successfully');
            document.getElementById('addCustomerForm').classList.add('hidden');
            document.querySelector('.header').style.visibility = 'visible';
            document.querySelector('.table__body').style.visibility = 'visible';
            fetchCustomers();
        } else {
            alert(`Error: ${data.message}`);
        }
    } catch (error) {
        console.error('Error adding customer:', error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    fetchCustomers();
    const searchInput = document.querySelector('.input-group input');
    searchInput.addEventListener('input', searchTable);
});


const search = document.querySelector(".input-group input")

search.addEventListener('input',searchTable);

	function searchTable(){
        let table_rows = document.querySelectorAll('tbody tr');
	    table_rows.forEach((row,i)=>{
	        let table_data = row.textContent.toLowerCase(),
	            search_data = search.value.toLowerCase ();
	            row.classList.toggle('hide',table_data.indexOf(search_data) <0 );
	            row.style.setProperty('--delay',i/25 + 's');
	    })
	}

async function viewCustomerDetails(customerId) {
    try {
        const response = await fetch(`${baseUrl}/information-management/information/account/${customerId}`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const customer = await response.json();
        document.querySelector('input[name="id_view"]').value = customer.id_account;
        document.querySelector('input[name="name_view"]').value = customer.name;
        document.querySelector('input[name="phone_view"]').value = customer.phone;
        document.querySelector('input[name="cccd_view"]').value = customer.cccd;
        document.querySelector('input[name="gmail_view"]').value = customer.gmail;
        document.querySelector(`input[name="gender_view"][value="${customer.gender ? 'Male' : 'Female'}"]`).checked = true;

        document.getElementById('detailCustomerForm').classList.remove('hidden');
        document.querySelector('.header').style.display = 'none';
        document.querySelector('.table__body').style.display = 'none';

    } catch (error) {
        console.error('Error fetching customer details:', error);
    }
}

function toggleAddCustomerForm() {
    document.getElementById('addCustomerForm').classList.remove('hidden');
    document.querySelector('.header').style.visibility = 'hidden';
    document.querySelector('.table__body').style.visibility = 'hidden';
}


async function handleAction(event, formElement, actione) {
    event.preventDefault();
    const formData = new FormData(formElement);
    const data = {
        id_customer: formData.get('id_view'),
        name: formData.get('name_view'),
        phone: formData.get('phone_view'),
        cccd: formData.get('cccd_view'),
        gmail: formData.get('gmail_view'),
        gender: formData.get('gender_view') === 'Male'
    };

    let method = 'PUT';
    let url = `${baseUrl}/information-management/information/account/${data.id_customer}`;

    if (actione === "Remove") {
        method = 'DELETE';
    }

    try {
        const response = await fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            },
            body: actione === "Remove" ? null : JSON.stringify(data),
        });

        const result = await response.json();

        if (response.ok) {
            if (actione === "Remove") {
                alert(`Delete successful`);
            } else {
                alert(`Update successful`);
            }
            document.getElementById('detailCustomerForm').classList.add('hidden');
            document.querySelector('.header').style.visibility = 'visible';
            document.querySelector('.table__body').style.visibility = 'visible';
            fetchCustomers();
        } else {
            document.getElementById('error-message').textContent = result.message
        }
    } catch (error) {
        document.getElementById('error-message').textContent = `Có lỗi xảy ra ` + error.message;
    }
}
