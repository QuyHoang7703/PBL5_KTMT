from .extension import ma 

class AccountSchema(ma.Schema):
    class Meta:
        fields = ('id_account', 'username', 'password', 'name', 'phone')

class CustomerSchema(ma.Schema):
    class Meta:
        fields = ('id_customer', 'name', 'gender', 'phone', 'cccd')

class HistorySchema(ma.Schema):
    class Meta:
        fields = ('id_history', 'vehicle_plate', 'date_in', 'date_out', 'time_in', 'time_out')

class TicketSchema(ma.Schema):
    class Meta:
        fields = ('vehicle_plate', 'init_date', 'expiry', 'id_customer')